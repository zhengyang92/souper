// Copyright 2014 The Souper Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "souper"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "souper/Extractor/Solver.h"
#include "souper/Infer/ExhaustiveSynthesis.h"
#include "souper/Infer/InstSynthesis.h"
#include "souper/KVStore/KVStore.h"
#include "souper/Parser/Parser.h"

#include <unordered_map>

STATISTIC(MemHitsInfer, "Number of internal cache hits for infer()");
STATISTIC(MemMissesInfer, "Number of internal cache misses for infer()");
STATISTIC(MemHitsIsValid, "Number of internal cache hits for isValid()");
STATISTIC(MemMissesIsValid, "Number of internal cache misses for isValid()");
STATISTIC(ExternalHits, "Number of external cache hits");
STATISTIC(ExternalMisses, "Number of external cache misses");

using namespace souper;
using namespace llvm;

namespace {

static cl::opt<bool> NoInfer("souper-no-infer",
    cl::desc("Populate the external cache, but don't infer replacements (default=false)"),
    cl::init(false));
static cl::opt<bool> InferNop("souper-infer-nop",
    cl::desc("Infer that the output is the same as an input value (default=false)"),
    cl::init(false));
static cl::opt<bool> StressNop("souper-stress-nop",
    cl::desc("stress-test big queries in nop synthesis by always performing all of the small queries (slow!) (default=false)"),
    cl::init(false));
static cl::opt<int>MaxNops("souper-max-nops",
    cl::desc("maximum number of values from the LHS to try to use as the RHS (default=20)"),
    cl::init(20));
static cl::opt<bool> InferInts("souper-infer-iN",
    cl::desc("Infer iN integers for N>1 (default=true)"),
    cl::init(true));
static cl::opt<bool> InferInsts("souper-infer-inst",
    cl::desc("Infer instructions (default=false)"),
    cl::init(false));
static cl::opt<bool> EnableExhaustiveSynthesis("souper-exhaustive-synthesis",
    cl::desc("Use exaustive search for instruction synthesis (default=false)"),
    cl::init(false));
static cl::opt<int> MaxLHSSize("souper-max-lhs-size",
    cl::desc("Max size of LHS (in bytes) to put in external cache (default=1024)"),
    cl::init(1024));

class BaseSolver : public Solver {
  std::unique_ptr<SMTLIBSolver> SMTSolver;
  unsigned Timeout;

public:
  BaseSolver(std::unique_ptr<SMTLIBSolver> SMTSolver, unsigned Timeout)
      : SMTSolver(std::move(SMTSolver)), Timeout(Timeout) {}

  bool testZeroSign(const BlockPCs &BPCs,
                const std::vector<InstMapping> &PCs,
                APInt &Negative, Inst *LHS,
                InstContext &IC) {
    unsigned W = LHS->Width;
    APInt Zero = APInt::getNullValue(W);
    Inst *Mask = IC.getConst(Negative);
    InstMapping Mapping(IC.getInst(Inst::And, W, { LHS, Mask }), IC.getConst(Zero));
    bool IsSat;
    Mapping.LHS->DemandedBits = APInt::getAllOnesValue(Mapping.LHS->Width);
    std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                  IsSat, 0, 0, Timeout);
    if (EC)
      llvm::report_fatal_error("stopping due to error");
    return !IsSat;
  }

  bool testOneSign(const BlockPCs &BPCs,
                const std::vector<InstMapping> &PCs,
                APInt &Negative, Inst *LHS,
                InstContext &IC) {
    unsigned W = LHS->Width;
    Inst *Mask = IC.getConst(Negative);
    InstMapping Mapping(IC.getInst(Inst::And, W, { LHS, Mask }), Mask);
    bool IsSat;
    Mapping.LHS->DemandedBits = APInt::getAllOnesValue(Mapping.LHS->Width);
    std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                  IsSat, 0, 0, Timeout);
    if (EC)
      llvm::report_fatal_error("stopping due to error");
    return !IsSat;
  }

  std::error_code Negative(const BlockPCs &BPCs,
                           const std::vector<InstMapping> &PCs,
                           Inst *LHS, APInt &Negative,
                           InstContext &IC) override {
    unsigned W = LHS->Width;
    Negative = APInt::getNullValue(W);
    APInt NegativeGuess = Negative | APInt::getOneBitSet(W, W-1);
    if (testZeroSign(BPCs, PCs, NegativeGuess, LHS, IC))
      Negative = APInt::getNullValue(W);
    else if (testOneSign(BPCs, PCs, NegativeGuess, LHS, IC))
      Negative = NegativeGuess;
    return std::error_code();
  }

  std::error_code nonNegative(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, APInt &NonNegative,
                              InstContext &IC) override {
    unsigned W = LHS->Width;
    NonNegative = APInt::getNullValue(W);
    APInt NonNegativeGuess = NonNegative | APInt::getOneBitSet(W, W-1);
    if (testZeroSign(BPCs, PCs, NonNegativeGuess, LHS, IC))
      NonNegative = APInt::getNullValue(W);
    else if (testOneSign(BPCs, PCs, NonNegativeGuess, LHS, IC))
      NonNegative = NonNegativeGuess;
    else
      NonNegative = NonNegativeGuess; //if sign-bit is not guessed as 0 or 1, set non-negative signbit to 1, so that nothing is inferred by souper at the end
    return std::error_code();
  }

  bool testKnown(const BlockPCs &BPCs,
                const std::vector<InstMapping> &PCs,
                APInt &Zeros, APInt &Ones, Inst *LHS,
                InstContext &IC) {
    unsigned W = LHS->Width;
    Inst *Mask = IC.getConst(Zeros | Ones);
    InstMapping Mapping(IC.getInst(Inst::And, W, { LHS, Mask }), IC.getConst(Ones));
    bool IsSat;
    Mapping.LHS->DemandedBits = APInt::getAllOnesValue(Mapping.LHS->Width);
    std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                  IsSat, 0, 0, Timeout);
    if (EC)
      llvm::report_fatal_error("stopping due to error");
    return !IsSat;
  }

  std::error_code knownBits(const BlockPCs &BPCs,
                          const std::vector<InstMapping> &PCs,
                          Inst *LHS, APInt &Zeros, APInt &Ones,
                          InstContext &IC) override {
    unsigned W = LHS->Width;
    Ones = APInt::getNullValue(W);
    Zeros = APInt::getNullValue(W);
    for (unsigned I=0; I<W; I++) {
      APInt ZeroGuess = Zeros | APInt::getOneBitSet(W, I);
      if (testKnown(BPCs, PCs, ZeroGuess, Ones, LHS, IC)) {
        Zeros = ZeroGuess;
        continue;
      }
      APInt OneGuess = Ones | APInt::getOneBitSet(W, I);
      if (testKnown(BPCs, PCs, Zeros, OneGuess, LHS, IC))
        Ones = OneGuess;
    }
    return std::error_code();
  }

  std::error_code powerTwo(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, APInt &PowTwo,
                              InstContext &IC) override {
    unsigned W = LHS->Width;
    APInt ConstOne(W, 1, false);
    APInt ZeroGuess(W, 0, false);
    APInt TrueGuess(1, 1, false);
    Inst *PowerMask = IC.getInst(Inst::And, W, {LHS, IC.getInst(Inst::Sub, W, {LHS, IC.getConst(ConstOne)})});
    Inst *Zero = IC.getConst(ZeroGuess);
    Inst *True = IC.getConst(TrueGuess);
    Inst *PowerTwoInst = IC.getInst(Inst::And, 1, {IC.getInst(Inst::Ne, 1, {LHS, Zero}),
                                                   IC.getInst(Inst::Eq, 1, {PowerMask, Zero})});
    InstMapping Mapping(PowerTwoInst, True);
    //InstMapping Mapping(PowerMask, Zero);
    bool IsSat;
    Mapping.LHS->DemandedBits = APInt::getAllOnesValue(Mapping.LHS->Width);
    std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                  IsSat, 0, 0, Timeout);
    if (EC)
      llvm::report_fatal_error("stopping due to error");

    if (!IsSat)
      PowTwo = APInt(1, 1, false);
    else
      PowTwo = APInt(1, 0, false);
    return std::error_code();
  }

  std::error_code nonZero(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, APInt &NonZero,
                              InstContext &IC) override {
    unsigned W = LHS->Width;
    APInt ZeroGuess(W, 0, false);
    APInt TrueGuess(1, 1, false);
    Inst *Zero = IC.getConst(ZeroGuess);
    Inst *True = IC.getConst(TrueGuess);
    Inst *NonZeroGuess = IC.getInst(Inst::Ne, 1, {LHS, Zero});
    InstMapping Mapping(NonZeroGuess, True);
    bool IsSat;
    Mapping.LHS->DemandedBits = APInt::getAllOnesValue(Mapping.LHS->Width);
    std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                  IsSat, 0, 0, Timeout);
    if (EC)
      llvm::report_fatal_error("stopping due to error");

    if (!IsSat)
      NonZero = APInt(1, 1, false);
    else
      NonZero = APInt(1, 0, false);
    return std::error_code();
  }

  std::error_code signBits(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, unsigned &SignBits,
                              InstContext &IC) override {
    unsigned W = LHS->Width;
    SignBits = 1;
    APInt Zero(W, 0, false);
    Inst *AllZeros = IC.getConst(Zero);
    APInt Ones = APInt::getAllOnesValue(W);
    Inst *AllOnes = IC.getConst(Ones);
    APInt TrueGuess(1, 1, false);
    Inst *True = IC.getConst(TrueGuess);
    // guess signbits starting from 2, because 1 is by default
    for (unsigned I=2; I<=W; I++) {
      APInt SA(W, W-I, false);
      Inst *ShiftAmt = IC.getConst(SA);
      Inst *Res = IC.getInst(Inst::AShr, W, {LHS, ShiftAmt}); 
      Inst *Guess1 = IC.getInst(Inst::Eq, 1, {Res, AllZeros});
      Inst *Guess2 = IC.getInst(Inst::Eq, 1, {Res, AllOnes});
      Inst *Guess = IC.getInst(Inst::Or, 1, {Guess1, Guess2}); 
      InstMapping Mapping(Guess, True);
      bool IsSat;
      Mapping.LHS->DemandedBits = APInt::getAllOnesValue(Mapping.LHS->Width);
      std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                    IsSat, 0, 0, Timeout);
      if (EC)
        llvm::report_fatal_error("stopping due to error");
  
      if (!IsSat) { //guess is correct, keep looking for more signbits
        SignBits = I;
        continue;
      } else {
        break;
      }
    }
    return std::error_code();
  }

  Inst * set_traverse_to_find_and_update_var(Inst *node, Inst *OrigLHS, Inst *prev, unsigned bitPos, InstContext &IC, unsigned idx) {
    Inst *root = node;
    if (node->K == Inst::Var) {
      llvm::outs() << "** found var in set_traversal **\n";
      unsigned VarWidth = node->Width;
      APInt SetBit = APInt::getOneBitSet(VarWidth, bitPos);
      Inst *SetMask = IC.getInst(Inst::Or, VarWidth, {node, IC.getConst(SetBit)}); //xxxx || 0001
     
      llvm::outs() << "- - - - - - plain traverse set mask only ---\n"; 
      plain_traverse(SetMask);

      node = SetMask;
      prev->Ops[idx] = node;

      return OrigLHS;
    }
    for (unsigned Op=0; Op<node->Ops.size(); ++Op) {
      set_traverse_to_find_and_update_var(node->Ops[Op], OrigLHS, node, bitPos, IC, Op);
    }

    return OrigLHS;
  }

  void plain_traverse(Inst *LHS) {
    if (!LHS) return;
    llvm::outs() << "Kind = " << Inst::getKindName(LHS->K) << ", Value = " << LHS->Val <<"\n";
    for (auto Op: LHS->Ops) {
      plain_traverse(Op);
    }
  }

  Inst * clear_traverse_to_find_and_update_var(Inst *node, Inst *OrigLHS, Inst *prev, unsigned bitPos, InstContext &IC, unsigned idx) {
    Inst *root = node;
    if (node->K == Inst::Var) {
      unsigned VarWidth = node->Width;

      APInt AllOnes = APInt::getAllOnesValue(VarWidth); //1111
      APInt ClearBit = getClearedBit(bitPos, VarWidth); //1110

      Inst *ClearMask = IC.getInst(Inst::And, VarWidth, {node, IC.getConst(ClearBit)}); // xxxx && ~(0001)

      llvm::outs() << "~~~~~~~ plain traverse just clear mask: \n";
      plain_traverse(ClearMask);

      node = ClearMask;
      prev->Ops[idx] = node;

      return OrigLHS;
    }
    for (unsigned Op=0; Op<node->Ops.size(); ++Op) {
      clear_traverse_to_find_and_update_var(node->Ops[Op], OrigLHS, node, bitPos, IC, Op);
    }

    return OrigLHS;
  }
  
  bool testDB(const BlockPCs &BPCs,
              const std::vector<InstMapping> &PCs,
              Inst *LHS, Inst *NewLHS,
              InstContext &IC) {
    unsigned W = LHS->Width;
//    Inst *Ne = IC.getInst(Inst::Ne, 1, {LHS, NewLHS});
//    APInt TrueGuess(1, 1, false);
//    Inst *True = IC.getConst(TrueGuess);
//    InstMapping Mapping(Ne, True);
    
    InstMapping Mapping(LHS, NewLHS);
    bool IsSat;
//    std::string Query = BuildQuery(IC, BPCs, PCs, Mapping, 0);
    std::string Query = BuildQuery(IC, BPCs, PCs, Mapping, 0, true);
    llvm::outs() << "==== Query ==== \n" << Query << "\n";
    std::error_code EC = SMTSolver->isSatisfiable(Query,
                                                  IsSat, 0, 0, Timeout);
    if (EC)
      llvm::report_fatal_error("stopping due to error");
    llvm::outs() << "Result of testDB = " << IsSat << "\n";
    return !IsSat;
  }

  llvm::APInt getClearedBit(unsigned Pos, unsigned W) {
    APInt AllOnes = APInt::getAllOnesValue(W);
    AllOnes.clearBit(Pos);
    return AllOnes;
  }

  std::error_code testDemandedBits(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, APInt &ResultDB,
                              InstContext &IC) override {
    unsigned W = LHS->Width;
    std::map<Inst *, Inst *> InstCache;
    std::map<Block *, Block *> BlockCache;
    Inst *CopyLHS = getInstCopy(LHS, IC, InstCache, BlockCache, 0, true);

    ResultDB = APInt::getNullValue(W);

    for (unsigned I=0; I<W; I++) {
      llvm::outs() << "=================== Bit = " << I << " ============\n";
      std::map<Inst *, Inst *> InstCache;
      std::map<Block *, Block *> BlockCache;
      Inst *OrigLHS1 = getInstCopy(CopyLHS, IC, InstCache, BlockCache, 0, true);
      Inst *SetLHS = set_traverse_to_find_and_update_var(OrigLHS1, OrigLHS1, OrigLHS1, I, IC, 0);

      llvm::outs() << "------- Set traversal tree is:\n";
      plain_traverse(SetLHS);

      std::map<Inst *, Inst *> InstCache2;
      std::map<Block *, Block *> BlockCache2;
      Inst *OrigLHS2 = getInstCopy(CopyLHS, IC, InstCache2, BlockCache2, 0, true);

      Inst *ClearLHS = clear_traverse_to_find_and_update_var(OrigLHS2, OrigLHS2, OrigLHS2, I, IC, 0);
      llvm::outs() << "******* Clear traversal tree is:\n";
      plain_traverse(ClearLHS);

      if (testDB(BPCs, PCs, CopyLHS, SetLHS, IC) && testDB(BPCs, PCs, CopyLHS, ClearLHS, IC)) {
        // not-demanded
        llvm::outs() << "Bit = " << I << " = not-demanded\n";
        ResultDB = ResultDB;
      } else {
        // demanded
        llvm::outs() << "Bit = " << I << " = demanded\n";
        ResultDB |= APInt::getOneBitSet(W, I);
      }
    }
    return std::error_code();
  }

//  std::error_code range(const BlockPCs &BPCs,
//                              const std::vector<InstMapping> &PCs,
//                              Inst *LHS, llvm::ConstantRange &Range,
//                              InstContext &IC) override {
//    unsigned W = LHS->Width;
//    APInt TrueGuess(1, 1, false);
//    Inst *True = IC.getConst(TrueGuess);
//    // trying only with non-wrapped range first, so lets initialize
//    // range with Int_MIN to INT_MAX
//    llvm::outs() << "signed max value = " << APInt::getSignedMaxValue(W) << "\n";
//    Range = llvm::ConstantRange(APInt(W, 0, false), APInt::getSignedMaxValue(W));
//    
//    // verify if LHS is between MIN and MAX - this should be true in all non-wrapped cases
//    Inst *Lower = IC.getConst(Range.getLower());
//    Inst *Upper = IC.getConst(Range.getUpper());
//
//    Inst *GuessLowerPart = IC.getInst(Inst::Ule, 1, {Lower, LHS});
//    Inst *GuessUpperPart = IC.getInst(Inst::Ult, 1, {LHS, Upper});
//    Inst *Guess = IC.getInst(Inst::And, 1, { GuessLowerPart, GuessUpperPart });
//
//    InstMapping Mapping(Guess, True);
//    bool IsSat;
//    std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
//                                                  IsSat, 0, 0, Timeout);
//    if (EC)
//      llvm::report_fatal_error("stopping due to error");
//  
//    if (!IsSat) {
//      //Guess is correct, keep looking for shorter range here later using binary search
//      continue;
//    }
//      Range = llvm::ConstantRange(APInt(W, 0, false), APInt::getSignedMaxValue(W));
//    else {
//      llvm::outs() << "full set range\n";
//      Range = llvm::ConstantRange(W, /*isFullSet*/true);
//    }
//    return std::error_code();
//  }

  std::error_code searchUpperBound(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, llvm::ConstantRange &Range,
                              APInt &PreviousUpper, APInt &NewUpper,
                              InstContext &IC) {
    unsigned W = LHS->Width;
    APInt TrueGuess(1, 1, false);
    Inst *True = IC.getConst(TrueGuess);

    APInt FinalUpper(W, 0);
    APInt One(W, 1);
    
    APInt Mid(W, 0);
    Mid += PreviousUpper;
    Mid += NewUpper;
    Mid = Mid.sdiv(APInt(W, 2));
    Inst *Guess = 0;

    APInt Diff(W, 0);
    Diff += PreviousUpper;
    Diff -= NewUpper;
    Diff = Diff.abs();

    bool IsGuessTrue;
    //llvm::outs() << "In searchUpperBound()\n";

    if (Diff.ule(APInt(W, 2))) {
      //llvm::outs() << "Diff between PU and NU is <= 2\n";
      Guess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(NewUpper)});
      InstMapping Mapping(Guess, True);

      std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                  IsGuessTrue, 0, 0, Timeout);
      if (EC)
        llvm::report_fatal_error("stopping due to error");

      if (!IsGuessTrue) {
        // x <= NewUpper
        // now test for x < NwUpper
        Guess = IC.getInst(Inst::Slt, 1, {LHS, IC.getConst(NewUpper)});
        InstMapping Mapping(Guess, True);

        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                    IsGuessTrue, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");

        if (!IsGuessTrue) {
          FinalUpper = NewUpper;
        } else {
          FinalUpper = NewUpper;
          FinalUpper += One;
        }
        //llvm::outs() << "LHS <= " << NewUpper << " = FinalUpper \n";
      } else {
        // if x <= mid?
        Guess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(Mid)});
        InstMapping Mapping(Guess, True);

        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsGuessTrue, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");

        if (!IsGuessTrue) {
          Guess = IC.getInst(Inst::Slt, 1, {LHS, IC.getConst(Mid)});
          InstMapping Mapping(Guess, True);

          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsGuessTrue, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");

          if (!IsGuessTrue) {
            FinalUpper = Mid;
          } else {
            FinalUpper = Mid;
            FinalUpper += One;
          }
          //llvm::outs() << "LHS <= " << Mid << " = FinalUpper \n";
        } else {
          // x <= PrevUpper
          Guess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(PreviousUpper)});
          InstMapping Mapping(Guess, True);

          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsGuessTrue, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");

          if (!IsGuessTrue) {
            Guess = IC.getInst(Inst::Slt, 1, {LHS, IC.getConst(PreviousUpper)});
            InstMapping Mapping(Guess, True);

            std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                        IsGuessTrue, 0, 0, Timeout);
            if (EC)
              llvm::report_fatal_error("stopping due to error");
            if (!IsGuessTrue) {
              FinalUpper = PreviousUpper;
            } else {
              FinalUpper = PreviousUpper;
              FinalUpper += One;
            }
          }
          //llvm::outs() << "LHS <= " << PreviousUpper << " = FinalUpper\n";
        }
      }
      Range = llvm::ConstantRange(Range.getLower(), FinalUpper);
      //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
    } else {
      // look if upper bound is on left or right of left
      //llvm::outs() << "Diff between NU and PU > 2\n";
      Guess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(Mid)});
      InstMapping Mapping(Guess, True);

      std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                    IsGuessTrue, 0, 0, Timeout);
      if (EC)
        llvm::report_fatal_error("stopping due to error");

      if (!IsGuessTrue) {
        //PreviousUpper = Mid;
        return searchUpperBound(BPCs, PCs, LHS, Range, Mid, NewUpper, IC);
      } else {
        //NewUpper = Mid;
        return searchUpperBound(BPCs, PCs, LHS, Range, PreviousUpper, Mid, IC);
      }
    }
    Range = llvm::ConstantRange(Range.getLower(), FinalUpper);
    //llvm::outs()  << "** just before return in searchUB: Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";

    return std::error_code();
  }

  std::error_code searchLowerBound(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, llvm::ConstantRange &Range,
                              APInt &PreviousLower, APInt &NewLower,
                              InstContext &IC) {
    unsigned W = LHS->Width;
    APInt TrueGuess(1, 1, false);
    Inst *True = IC.getConst(TrueGuess);

    APInt FinalLower(W, 0);
    APInt One(W, 1);
    
    APInt Mid(W, 0);
    Mid += PreviousLower;
    Mid += NewLower;
    Mid = Mid.sdiv(APInt(W, 2));
    Inst *Guess = 0;

    APInt Diff(W, 0);
    Diff += PreviousLower;
    Diff -= NewLower;
    Diff = Diff.abs();

    bool IsGuessTrue;

    if (Diff.ule(APInt(W, 2))) {
      // diff is less than two, then look for three low values directly
      Guess = IC.getInst(Inst::Sle, 1, {IC.getConst(NewLower), LHS});
      InstMapping Mapping(Guess, True);

      std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                  IsGuessTrue, 0, 0, Timeout);
      if (EC)
        llvm::report_fatal_error("stopping due to error");

      if (!IsGuessTrue) {
        // x >= NL
        // now test for x>NL
        Guess = IC.getInst(Inst::Slt, 1, {IC.getConst(NewLower), LHS});
        InstMapping Mapping(Guess, True);

        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                    IsGuessTrue, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");

        if (!IsGuessTrue) {
          FinalLower = NewLower;
          FinalLower += One;
        } else {
          FinalLower = NewLower;
        }
        //llvm::outs() << "LHS <= " << NewLower << " = FinalLower \n";
      } else {
        // if x >= mid?
        Guess = IC.getInst(Inst::Sle, 1, {IC.getConst(Mid), LHS});
        InstMapping Mapping(Guess, True);

        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsGuessTrue, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");

        if (!IsGuessTrue) {
          Guess = IC.getInst(Inst::Slt, 1, {IC.getConst(Mid), LHS});
          InstMapping Mapping(Guess, True);

          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsGuessTrue, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");

          if (!IsGuessTrue) {
            // x > Mid => finalLow = mid + 1
            FinalLower = Mid;
            FinalLower += One;
          } else {
            FinalLower = Mid;
          }
          //llvm::outs() << "LHS <= " << Mid << " = FinalLower \n";
        } else {
          // x >= PrevLower
          Guess = IC.getInst(Inst::Sle, 1, {IC.getConst(PreviousLower), LHS});
          InstMapping Mapping(Guess, True);

          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsGuessTrue, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");

          if (!IsGuessTrue) {
            // now check if x > prevlower
            Guess = IC.getInst(Inst::Slt, 1, {IC.getConst(PreviousLower), LHS});
            InstMapping Mapping(Guess, True);

            std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                        IsGuessTrue, 0, 0, Timeout);
            if (EC)
              llvm::report_fatal_error("stopping due to error");
            if (!IsGuessTrue) {
              FinalLower = PreviousLower;
              FinalLower += One;
            } else {
              FinalLower = PreviousLower;
            }
          }
        }
      }

//      if (!IsGuessTrue) {
//        // x >= new lower
//        FinalLower = NewLower;
//      } else {
//        // if x >= mid?
//        Guess = IC.getInst(Inst::Sle, 1, {IC.getConst(Mid), LHS});
//        InstMapping Mapping(Guess, True);
//
//        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
//                                                      IsGuessTrue, 0, 0, Timeout);
//        if (EC)
//          llvm::report_fatal_error("stopping due to error");
//
//        if (!IsGuessTrue) {
//          FinalLower = Mid;
//        } else {
//          // x >= PrevLower
//          FinalLower = PreviousLower;
//        }
      Range = llvm::ConstantRange(FinalLower, Range.getUpper());
      //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
    } else {
      // look if lower bound is on left or right of left
      Guess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(Mid)});
      InstMapping Mapping(Guess, True);

      std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                    IsGuessTrue, 0, 0, Timeout);
      if (EC)
        llvm::report_fatal_error("stopping due to error");

      if (!IsGuessTrue) {
        //NewLower = Mid;
        return searchLowerBound(BPCs, PCs, LHS, Range, PreviousLower, Mid, IC);
      } else {
        //PreviousLower = Mid;
        return searchLowerBound(BPCs, PCs, LHS, Range, Mid, NewLower, IC);
      }
    }

    return std::error_code();
  }

  std::error_code moreRangeTest(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS,
                              llvm::ConstantRange &Range,
                              llvm::APInt &PreviousLower,
                              llvm::APInt &PreviousUpper,
                              llvm::APInt &NewLower, llvm::APInt &NewUpper,
                              InstContext &IC) {
    unsigned W = LHS->Width;
    //llvm::outs() << "In func: moreRangeTest()\n";
    //llvm::outs() << "ENTRY of G_FN: Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
    //llvm::outs() << "PL, PU = " << PreviousLower << ", " << PreviousUpper << "\n";
    //llvm::outs() << "NL, NU = " << NewLower << ", " << NewUpper << "\n";
    //llvm::outs() << "Difference of prev low and new low\n";

    APInt DiffLower(W, 0);
    DiffLower += PreviousLower;
    DiffLower -= NewLower;
    DiffLower = DiffLower.abs();
    //llvm::outs() << "Diff Lower = " << DiffLower << "\n";

    APInt DiffUpper(W, 0);
    DiffUpper += PreviousUpper;
    DiffUpper -= NewUpper;
    DiffUpper = DiffUpper.abs();
    //llvm::outs() << "Diff Upper = " << DiffUpper << "\n";

    APInt FinalLower(W, 0);
    APInt FinalUpper(W, 0);
    APInt One(W, 1);
    if (DiffLower.ule(APInt(W, 1)) && DiffUpper.ule(APInt(W, 1))) {
      // simple check and return the answer
      //llvm::outs() << "Case: difflow <= 1 and diffup <= 1\n";
      if (NewLower.sgt(PreviousLower)) { // new > prev, test x>=new
        Inst *FinalLValue = IC.getConst(NewLower);
        Inst *Guess = IC.getInst(Inst::Sle, 1, {IC.getConst(NewLower), LHS});
        APInt TrueGuess(1, 1, false);
        Inst *True = IC.getConst(TrueGuess);

        InstMapping Mapping(Guess, True);
        bool IsSat;
        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsSat, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");
        if (!IsSat) {
          // x >= NL
          // but, see if x > NL only
          Guess = IC.getInst(Inst::Slt, 1, {IC.getConst(NewLower), LHS});
          InstMapping Mapping(Guess, True);

          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                        IsSat, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");

          if (!IsSat) {
            FinalLower = NewLower;
            FinalLower += One;
          } else {
            FinalLower = NewLower;
          }
          //llvm::outs() << "Final Lower == " << FinalLower << "\n";
        } else {
          // final lower = previous lower
          Guess = IC.getInst(Inst::Sle, 1, {IC.getConst(PreviousLower), LHS});
          InstMapping Mapping(Guess, True);
          
          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                        IsSat, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");

          if (!IsSat) {
            Guess = IC.getInst(Inst::Slt, 1, {IC.getConst(PreviousLower), LHS});
            InstMapping Mapping(Guess, True);
            
            std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                          IsSat, 0, 0, Timeout);
            if (EC)
              llvm::report_fatal_error("stopping due to error");

            if (!IsSat) {
              FinalLower = PreviousLower;
              FinalLower += One;
            } else {
              FinalLower = PreviousLower;
            }
          }
          //llvm::outs() << "Final Lower == " << FinalLower << "\n";
        }
      }
      // finalize the upper bound
      if (NewUpper.slt(PreviousUpper)) { // new < prev, test x<=new
        Inst *FinalLValue = IC.getConst(NewUpper);
        Inst *Guess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(NewUpper)});
        APInt TrueGuess(1, 1, false);
        Inst *True = IC.getConst(TrueGuess);

        InstMapping Mapping(Guess, True);
        bool IsSat;
        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsSat, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");
        if (!IsSat) {
          // x<= NU, but look if x < NU only?

          Guess = IC.getInst(Inst::Slt, 1, {LHS, IC.getConst(NewUpper)});
          InstMapping Mapping(Guess, True);

          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                        IsSat, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");

          if (!IsSat) {
            FinalUpper = NewUpper;
          } else {
            FinalUpper = NewUpper;
            FinalUpper += One;
          }
          //llvm::outs() << "Final Upper == " << FinalUpper << "\n";
        } else {
          // final lower = previous lower

          Guess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(PreviousUpper)});
          InstMapping Mapping(Guess, True);
          
          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                        IsSat, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");

          if (!IsSat) {
            Guess = IC.getInst(Inst::Slt, 1, {LHS, IC.getConst(PreviousUpper)});
            InstMapping Mapping(Guess, True);
            
            std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                          IsSat, 0, 0, Timeout);
            if (EC)
              llvm::report_fatal_error("stopping due to error");

            if (!IsSat) {
              FinalUpper = PreviousUpper;
            } else {
              FinalUpper = PreviousUpper;
              FinalUpper += One;
            }
          }
          //llvm::outs() << "Final Upper == " << FinalUpper << "\n";
        }
      }
      Range = llvm::ConstantRange(FinalLower, FinalUpper);
      //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
    } else if (DiffLower.ugt(APInt(W, 1)) && DiffUpper.ugt(APInt(W, 1))) {
      // keep iterating as normal
      //llvm::outs() << "case: diff low > 1 and diff up > 1 -- make three ranges for testing here!!\n";
      if (testRange(BPCs, PCs, LHS, NewLower, NewUpper, IC)) {
        Range = llvm::ConstantRange(NewLower, NewUpper);
        //llvm::outs() << "Test passed for: Low = " << NewLower << ",  Up = " << NewUpper << "\n";
        return range(BPCs, PCs, LHS, Range, NewLower, NewUpper, IC);
      } else {
        if (testRange(BPCs, PCs, LHS, NewLower, PreviousUpper, IC)) {
          Range = llvm::ConstantRange(NewLower, PreviousUpper);
          //llvm::outs() << "Test passed for  low = " << NewLower << ", Upper = " << PreviousUpper << "\n";
          return range(BPCs, PCs, LHS, Range, NewLower, PreviousUpper, IC);
        } else {
          if (testRange(BPCs, PCs, LHS, PreviousLower, NewUpper, IC)) {
            //llvm::outs() << "Test passed for low = " << PreviousLower << ", " << NewUpper << "\n";
            Range = llvm::ConstantRange(PreviousLower, NewUpper);
            return range(BPCs, PCs, LHS, Range, PreviousLower, NewUpper, IC);
          } else {
            // we learned another new lower and upper range
            // new lower mid = pl + nl / 2
            // new upper mid = nu + pu / 2
            //llvm::outs() << "No mini tests passed, learn and iterate again\n\n";
            APInt NewLowerMid(W, 0);
            NewLowerMid += PreviousLower;
            NewLowerMid += NewLower;
            NewLowerMid = NewLowerMid.sdiv(APInt(W, 2));

            APInt NewUpperMid(W, 0);
            NewUpperMid += PreviousUpper;
            NewUpperMid += NewUpper;
            NewUpperMid = NewUpperMid.sdiv(APInt(W, 2));
            //llvm::outs() << "New L = " << NewLowerMid << ", New U = " << NewUpperMid << "\n";
            Range = llvm::ConstantRange(PreviousLower, PreviousUpper);
            //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
            return moreRangeTest(BPCs, PCs, LHS, Range, PreviousLower, PreviousUpper, NewLowerMid, NewUpperMid, IC);
          }
        }
      }
    } else if (DiffLower.ule(APInt(W, 1)) && DiffUpper.ugt(APInt(W, 1))) {
      // fix lower only iterate for upper
      //llvm::outs() << "case: diff low <= 1 and diff up > 1\n";
      if (NewLower.sgt(PreviousLower)) { // new > prev, test x>=new
        //llvm::outs() << "NL > PL\n";
        Inst *FinalLValue = IC.getConst(NewLower);
        Inst *Guess = IC.getInst(Inst::Sle, 1, {IC.getConst(NewLower), LHS});
        APInt TrueGuess(1, 1, false);
        Inst *True = IC.getConst(TrueGuess);

        InstMapping Mapping(Guess, True);
        bool IsSat;
        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsSat, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");
        if (!IsSat) {
          // guess is correct, means: lower final value is found equal to newlower
          //llvm::outs() << "LHS is always >= NL " << NewLower << "\n";
          DiffLower = APInt(W, 0);
          FinalLower = NewLower;
        } else {
          // final lower = previous lower
          //llvm::outs() << "LHS is always <=  PL " << PreviousLower << "\n";
          DiffLower = APInt(W, 0);
          FinalLower = PreviousLower;
        }
      }
      // iterate for upper now between newupper and previous upper only
      //llvm::outs() << "iterate for upper bound only \n\n";
      Range = llvm::ConstantRange(FinalLower, PreviousUpper);
      //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
      return searchUpperBound(BPCs, PCs, LHS, Range, PreviousUpper, NewUpper, IC);
    } else { // difflower > 1 and diffupper <= 1
      // fix the upper and only iterate for lower
      //llvm::outs() << "diff low > 1 and diff up <=1\n";
      if (NewUpper.slt(PreviousUpper)) { // new < prev, test x<=new
        Inst *FinalLValue = IC.getConst(NewUpper);
        Inst *Guess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(NewUpper)});
        APInt TrueGuess(1, 1, false);
        Inst *True = IC.getConst(TrueGuess);

        InstMapping Mapping(Guess, True);
        bool IsSat;
        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsSat, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");
        if (!IsSat) {
          // guess is correct, means: lower final value is found equal to newlower
          // x <= NU
          //llvm::outs() << "LHS is always <= NU " << NewUpper << "\n";
          DiffUpper = APInt(W, 0);
          FinalUpper = NewUpper;
        } else {
          // final lower = previous lower
          //llvm::outs() << "LHS is always <= PU " << PreviousUpper << "\n";
          DiffUpper = APInt(W, 0);
          FinalUpper = PreviousUpper;
        }
      }
      Range = llvm::ConstantRange(PreviousLower, FinalUpper);
      //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
      //iterate for lower bound only between prevlower and new lower
      //llvm::outs() << "Iterate for lower bound only\n\n";
      return searchLowerBound(BPCs, PCs, LHS, Range, PreviousLower, NewLower, IC);
    }

//    if (testRange(BPCs, PCs, LHS, NewLower, NewUpper, IC)) {
//      llvm::ConstantRange NewRange(NewLower, NewUpper);
//      llvm::outs() << "Test passed for: Low = " << NewLower << ",  Up = " << NewUpper << "\n";
//      return range(BPCs, PCs, LHS, NewRange, NewLower, NewUpper, IC);
//    } else {
//      if (testRange(BPCs, PCs, LHS, NewLower, PreviousUpper, IC)) {
//        llvm::ConstantRange NewRange(NewLower, PreviousUpper);
//        llvm::outs() << "Test passed for  low = " << NewLower << ", Upper = " << PreviousUpper << "\n";
//        return range(BPCs, PCs, LHS, NewRange, NewLower, PreviousUpper, IC);
//      } else {
//        if (testRange(BPCs, PCs, LHS, PreviousLower, NewUpper, IC)) {
//          llvm::outs() << "Test passed for low = " << PreviousLower << ", " << NewUpper << "\n";
//          llvm::ConstantRange NewRange(PreviousLower, NewUpper);
//          return range(BPCs, PCs, LHS, NewRange, PreviousLower, NewUpper, IC);
//        } else {
//          // we learned another new lower and upper range
//          // new lower mid = pl + nl / 2
//          // new upper mid = nu + pu / 2
//          llvm::outs() << "No mini tests passed, learn and iterate again\n\n";
//          APInt NewLowerMid(W, 0);
//          NewLowerMid += PreviousLower;
//          NewLowerMid += NewLower;
//          NewLowerMid = NewLowerMid.udiv(APInt(W, 2));
//
//          APInt NewUpperMid(W, 0);
//          NewUpperMid += PreviousUpper;
//          NewUpperMid += NewUpper;
//          NewUpperMid = NewUpperMid.udiv(APInt(W, 2));
//          llvm::outs() << "New L = " << NewLowerMid << ", New U = " << NewUpperMid << "\n";
//
//          return moreRangeTest(BPCs, PCs, LHS, PreviousLower, PreviousUpper, NewLowerMid, NewUpperMid, IC);
//        }
//      }
//    }
    return std::error_code();
  }

  bool testRange(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, llvm::APInt &Lower,
                              llvm::APInt &Upper,
                              InstContext &IC) {
    unsigned W = LHS->Width;

    Inst *LowerVal = IC.getConst(Lower);
    Inst *UpperVal = IC.getConst(Upper);

    Inst *GuessLowerPart = IC.getInst(Inst::Ule, 1, {LowerVal, LHS});
    Inst *GuessUpperPart = IC.getInst(Inst::Ult, 1, {LHS, UpperVal});
    Inst *Guess = 0;
    if (Upper.sgt(Lower)) { // U>L - not wrapped set
      Guess = IC.getInst(Inst::And, 1, { GuessLowerPart, GuessUpperPart });
    } else {
      Guess = IC.getInst(Inst::Or, 1, { GuessLowerPart, GuessUpperPart });
    }
    //if (Lower.sgt(Upper)) { // wrapped set
    //  Guess = IC.getInst(Inst::Or, 1, { GuessLowerPart, GuessUpperPart });
    //} else {
    //  Guess = IC.getInst(Inst::And, 1, { GuessLowerPart, GuessUpperPart });
    //}

    APInt TrueGuess(1, 1, false);
    Inst *True = IC.getConst(TrueGuess);

    InstMapping Mapping(Guess, True);
    bool IsSat;
    std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                  IsSat, 0, 0, Timeout);
    if (EC)
      llvm::report_fatal_error("stopping due to error");

    return !IsSat;
  }

  std::error_code range(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, llvm::ConstantRange &Range,
                              APInt &PreviousLow, APInt &PreviousUp,
                              InstContext &IC) override {
    unsigned W = LHS->Width;
    APInt TrueGuess(1, 1, false);
    Inst *True = IC.getConst(TrueGuess);
   
    APInt Lower = Range.getLower();
    APInt Upper = Range.getUpper();
    //llvm::outs() << "START range() : Lower = " << Lower << ", Upper = " << Upper << "\n";
    //llvm::outs() << "Start prev low = " << PreviousLow << ", prev up = " << PreviousUp << "\n";

//    llvm::outs() << "Try const synthesis here\n";
//    std::vector<Inst *> ModelInsts;
//    std::vector<llvm::APInt> ModelVals;
//    Inst *I = IC.createVar(LHS->Width, "constant");
//    Inst *LHSNeConst = IC.getInst(Inst::Ne, 1, {LHS, I});
//    InstMapping Mapping(LHSNeConst, True);
//    std::string Query = BuildQuery(IC, BPCs, PCs, Mapping, &ModelInsts, /*Negate=*/true);
//    if (Query.empty())
//      return std::make_error_code(std::errc::value_too_large);
//    bool IsSat;
//    std::error_code EC2 = SMTSolver->isSatisfiable(Query, IsSat, ModelInsts.size(),
//                                  &ModelVals, Timeout);
//    if (EC2)
//      return EC2;
//    if (IsSat) {
//      // We found a model for a constant
//      llvm::outs() << "SAT: found a model const\n";
//      Inst *Const = 0;
//      for (unsigned J = 0; J != ModelInsts.size(); ++J) {
//        if (ModelInsts[J]->Name == "constant") {
//          llvm::outs() << "for J = " << J << ", const val = " << ModelVals[J] << "\n";
//          Const = IC.getConst(ModelVals[J]);
//          break;
//        }
//      }
//      if (!Const)
//        report_fatal_error("there must be a model for the constant");
//
//      // Check if the constant is valid for all inputs
//      Inst *ForAll = IC.getInst(Inst::Ne, 1, {LHS, Const});
//      InstMapping ConstMapping(ForAll, True);
//      std::string Query = BuildQuery(IC, BPCs, PCs, ConstMapping, 0);
//      if (Query.empty())
//        return std::make_error_code(std::errc::value_too_large);
//      EC2 = SMTSolver->isSatisfiable(Query, IsSat, 0, 0, Timeout);
//      if (EC2)
//        return EC2;
//      if (!IsSat) {
//        //RHS = Const;
//        llvm::outs() << "Final const found = " << Const->Val << "\n";
//        return EC2;
//      }
//    }
      // We found a model for a constant

    bool IsGuessTrue;
    Inst *ZeroGuess = 0;
    if (Range.isFullSet()) {
      //llvm::outs() << "Full set testing --- w,r,t, 0\n";
      ZeroGuess = IC.getInst(Inst::Slt, 1, {LHS, IC.getConst(APInt(W, 0))});
      InstMapping Mapping(ZeroGuess, True);
  
      std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                    IsGuessTrue, 0, 0, Timeout);
      if (EC)
        llvm::report_fatal_error("stopping due to error");
  
      if (!IsGuessTrue) {
        // range is: MIN, 0
        //llvm::outs() << "x < 0\n";
        Range = llvm::ConstantRange(APInt::getSignedMinValue(W), APInt(W, 0));
        //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
        PreviousLow = APInt::getSignedMinValue(W);
        PreviousUp = APInt(W, 0);
        //llvm::outs() << "Low = PrevLow = " << PreviousLow << ", Up = PrevUp = " << PreviousUp << "\n";
      } else {
        // query SMT solver for x <= 0 and more cases
        ZeroGuess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(APInt(W, 0))});
        InstMapping Mapping(ZeroGuess, True);
  
        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsGuessTrue, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");
  
        if (!IsGuessTrue) {
          // range is: MIN, 1
          //llvm::outs() << "x <= 0\n";
          Range = llvm::ConstantRange(APInt::getSignedMinValue(W), APInt(W, 1));
          //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
          PreviousLow = APInt::getSignedMinValue(W);
          PreviousUp = APInt(W, 1);
          //llvm::outs() << "Low = PrevLow = " << PreviousLow << ", Up = PrevUp = " << PreviousUp << "\n";
        } else {
          // query SMT solver for x > 0 and more cases
          ZeroGuess = IC.getInst(Inst::Slt, 1, {IC.getConst(APInt(W, 0)), LHS});
          InstMapping Mapping(ZeroGuess, True);
  
          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                        IsGuessTrue, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");
  
          if (!IsGuessTrue) {
            // range is: 1, MIN
            //llvm::outs() << "x > 0\n";
            Range = llvm::ConstantRange(APInt(W, 1), APInt::getSignedMinValue(W));
            //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
            PreviousLow = APInt(W, 1);
            PreviousUp = APInt::getSignedMinValue(W);
            //llvm::outs() << "Low = PrevLow = " << PreviousLow << ", Up = PrevUp = " << PreviousUp << "\n";
          } else {
            // query SMT solver for x >= 0 and more cases
            ZeroGuess = IC.getInst(Inst::Sle, 1, {IC.getConst(APInt(W, 0)), LHS});
            InstMapping Mapping(ZeroGuess, True);
  
            std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                          IsGuessTrue, 0, 0, Timeout);
            if (EC)
              llvm::report_fatal_error("stopping due to error");
  
            if (!IsGuessTrue) {
              // range is: 0, MIN
              //llvm::outs() << "x >= 0\n";
              Range = llvm::ConstantRange(APInt(W, 0), APInt::getSignedMinValue(W));
              //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
              PreviousLow = APInt(W, 0);
              PreviousUp = APInt::getSignedMinValue(W);
              //llvm::outs() << "Low = PrevLow = " << PreviousLow << ", Up = PrevUp = " << PreviousUp << "\n";
            } else {
              // verify another special case if x != 0
              ZeroGuess = IC.getInst(Inst::Ne, 1, {IC.getConst(APInt(W, 0)), LHS});
              InstMapping Mapping(ZeroGuess, True);
  
              std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                            IsGuessTrue, 0, 0, Timeout);
              if (EC)
                llvm::report_fatal_error("stopping due to error");
  
              if (!IsGuessTrue) {
                // range is: 1,0
                //llvm::outs() << "x != 0\n";
                Range = llvm::ConstantRange(APInt(W, 1), APInt(W, 0));
                //llvm::outs() << "** Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
                PreviousLow = APInt(W, 1);
                PreviousUp = APInt(W, 0);
                //llvm::outs() << "Low = PrevLow = " << PreviousLow << ", Up = PrevUp = " << PreviousUp << "\n";
              } else {
                Range = llvm::ConstantRange(PreviousLow, PreviousUp);
                // x is both on +ve and -ve side of the numberline
                //APInt Low = APInt::getSignedMinValue(W);
                //APInt Up = APInt::getSignedMaxValue(W);
                //llvm::outs() << "In else case to test if Min to Max is working? with Low = " << Low << ", Up = " << Up << "\n";
                //APInt M(W, 0);
                //M = Up.sdiv(APInt(W, 2));
                //llvm::outs() << "Temp. UpperM = " << M << "\n";
                //if (testRange(BPCs, PCs, LHS, Low, Up, IC)) {
                //  llvm::outs() << "test range passed for smin to smax\n";
                //  PreviousLow = Low;
                //  PreviousUp = Up;
                //  Range = llvm::ConstantRange(PreviousLow, PreviousUp);
                //  APInt NewLowerMid = PreviousLow.sdiv(APInt(W, 2));
                //  APInt NewUpperMid = PreviousUp.sdiv(APInt(W, 2));
                //  return moreRangeTest(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, NewLowerMid, NewUpperMid, IC);
                //} else {
                //  Range = llvm::ConstantRange(PreviousLow, PreviousUp);
                //}
              }
            }
          }
        }
      }
    }
    
    //llvm::outs() << "Now TestRange() begins here for Low == " << Range.getLower() << ", Up = " << Range.getUpper() << "\n";
    APInt Mid = APInt(W, 0);
    APInt LowerVal = Range.getLower();
    APInt UpperVal = Range.getUpper();
    if (testRange(BPCs, PCs, LHS, LowerVal, UpperVal, IC)) {
      //llvm::outs() << "test range passed, so partition with Mid value further \n";
      // partition the range into two halves by verifying the mid value
      // if range.isNOTWrappedSet =>
      //   mid = low + up /2
      // else range.isWrappedSet =>
      //   distance_from_max = abs(lower - MAX)
      //   distance_from_min = abs(smin - upper)
      //   mid = total_distance/2
      //   range1 = lower, lower+mid 
      //   range2 = lower+mid, high
      Inst *MidGuess = 0;
      if (Range.getUpper().sgt(Range.getLower())) {
        // Lower s< upper
        Mid += Range.getLower();
        //llvm::outs() << "Mid = Lower = " << Mid << "\n";
        Mid += Range.getUpper();
        //llvm::outs() << "Mid = Lower + Upper = " << Mid << "\n";
        Mid = Mid.sdiv(APInt(W, 2));
        //llvm::outs() << "Mid = " << Mid << "\n";
      } else {
        // wrapped set (Lower is close to max, upper is close to min)
        APInt DistFromMax = APInt(W, 0);
        APInt DistFromMin = APInt(W, 0);
        APInt TotalElementsInSet = APInt(W, 0);
        DistFromMax += Range.getLower();
        DistFromMax -= APInt::getSignedMaxValue(W);
        TotalElementsInSet += DistFromMin.abs();
        TotalElementsInSet += DistFromMax.abs();
        Mid = TotalElementsInSet.sdiv(APInt(W, 2));
        Mid += Range.getLower();
        //llvm::outs() << "Mid = " << Mid << "\n";
      }
      MidGuess = IC.getInst(Inst::Slt, 1, {LHS, IC.getConst(Mid)});
      InstMapping Mapping(MidGuess, True);
  
      std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                    IsGuessTrue, 0, 0, Timeout);
      if (EC)
        llvm::report_fatal_error("stopping due to error");
  
      if (!IsGuessTrue) {
        // range is: lower, mid
        Range = llvm::ConstantRange(Range.getLower(), Mid);
        PreviousLow = Range.getLower();
        PreviousUp = Mid;
        //FIXME: IMP: TODO: check distance between range.lower and range.upper
        // if distance <=2 -> look for independent values or ranges in parts
        // return Range at the end

        APInt Diff(W, 0);
        Diff = PreviousUp;
        Diff -= PreviousLow;
        Diff = Diff.abs();

        if (Diff.ule(APInt(W, 2))) {
          // FIXME: IMP: TODO: Check the exact value of upper bound <= or < previousUp,
          // or check with <= or < Mid value first?
          Range = llvm::ConstantRange(PreviousLow, PreviousUp);
        } else {
          return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
        }
        //return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
      } else {
        MidGuess = IC.getInst(Inst::Sle, 1, {LHS, IC.getConst(Mid)});
        InstMapping Mapping(MidGuess, True);
        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                      IsGuessTrue, 0, 0, Timeout);
        if (EC)
          llvm::report_fatal_error("stopping due to error");
  
        if (!IsGuessTrue) {
          // range is: lower, mid+1
          PreviousLow = Range.getLower();
          APInt MidPlusOne(W, 1);
          MidPlusOne += Mid;
          PreviousUp = MidPlusOne;
          Range = llvm::ConstantRange(PreviousLow, PreviousUp);


          APInt Diff(W, 0);
          Diff = PreviousUp;
          Diff -= PreviousLow;
          Diff = Diff.abs();

          if (Diff.ule(APInt(W, 2))) {
            // FIXME: IMP: TODO: Check the exact value of upper bound <= or < previousUp or check with <= or < Mid value first?
            Range = llvm::ConstantRange(PreviousLow, PreviousUp);
          } else {
            return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
          }

          //return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
        } else {
          // make another guess w.r.t. mid value
          MidGuess = IC.getInst(Inst::Slt, 1, {IC.getConst(Mid), LHS});
          InstMapping Mapping(MidGuess, True);
          std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                        IsGuessTrue, 0, 0, Timeout);
          if (EC)
            llvm::report_fatal_error("stopping due to error");
  
          if (!IsGuessTrue) {
            // range is: mid+1, upper
            PreviousUp = Range.getUpper();
            APInt MidPlusOne(W, 1);
            MidPlusOne += Mid;
            PreviousLow = MidPlusOne;
            Range = llvm::ConstantRange(PreviousLow, PreviousUp);


            APInt Diff(W, 0);
            Diff = PreviousUp;
            Diff -= PreviousLow;
            Diff = Diff.abs();

            if (Diff.ule(APInt(W, 2))) {
              // FIXME: IMP: TODO: Check the exact value of upper bound <= or < previousUp or check with <= or < Mid value first?
              Range = llvm::ConstantRange(PreviousLow, PreviousUp);
            } else {
              return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
            }
            //return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
          } else {
            // make another guess
            MidGuess = IC.getInst(Inst::Sle, 1, {IC.getConst(Mid), LHS});
            InstMapping Mapping(MidGuess, True);
            std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
                                                          IsGuessTrue, 0, 0, Timeout);
            if (EC)
              llvm::report_fatal_error("stopping due to error");
  
            if (!IsGuessTrue) {
              // range is: mid, upper
              PreviousUp = Range.getUpper();
              PreviousLow = Mid;
              Range = llvm::ConstantRange(PreviousLow, PreviousUp);

              APInt Diff(W, 0);
              Diff = PreviousUp;
              Diff -= PreviousLow;
              Diff = Diff.abs();

              if (Diff.ule(APInt(W, 2))) {
                // FIXME: IMP: TODO: Check the exact value of upper bound <= or < previousUp or check with <= or < Mid value first?
                Range = llvm::ConstantRange(PreviousLow, PreviousUp);
              } else {
                return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
              }
              //return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
            } else {
              // TODO: Is this correct?
              //llvm::outs() << "******* Jubi: More opportunities to optimize *******\n";
              //llvm::outs() << "Mid value failed at: " << Mid << "\n";
              //llvm::outs() << "Previous success range is: " << PreviousLow << ", " << PreviousUp << "\n";
              APInt NewLowerMid = APInt(W, 0);
              NewLowerMid += PreviousLow;
              NewLowerMid += Mid;
              NewLowerMid = NewLowerMid.sdiv(APInt(W, 2));
              //llvm::outs() << "New lower mid partition is at: " << NewLowerMid << "\n";
              APInt NewUpperMid = APInt(W, 0);
              NewUpperMid += PreviousUp;
              NewUpperMid += Mid;
              NewUpperMid = NewUpperMid.sdiv(APInt(W, 2));
              //llvm::outs() << "New upper mid partition is at: " << NewUpperMid << "\n";
              //llvm::outs() << "test mini ranges now\n";
              //llvm::outs() << "Range = " << Range.getLower() << ", " << Range.getUpper() << "\n";
              return moreRangeTest(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, NewLowerMid, NewUpperMid, IC);
              //Range = llvm::ConstantRange(PreviousLow, PreviousUp);
            }
          }
        }
      }
    } else {
      // TODO: Deal with it more precisely later
      Range = llvm::ConstantRange(PreviousLow, PreviousUp);
    }

    return std::error_code();
  }

//  std::error_code range(const BlockPCs &BPCs,
//                              const std::vector<InstMapping> &PCs,
//                              Inst *LHS, llvm::ConstantRange &Range,
//                              APInt &PreviousLow, APInt &PreviousUp,
//                              InstContext &IC) override {
//    unsigned W = LHS->Width;
//    APInt TrueGuess(1, 1, false);
//    Inst *True = IC.getConst(TrueGuess);
//   
//    APInt Lower = Range.getLower();
//    APInt Upper = Range.getUpper();
//    llvm::outs() << "range() : Lower = " << Lower << ", Upper = " << Upper << "\n";
//
//    if (testRange(BPCs, PCs, LHS, Lower, Upper, IC) || Range.isFullSet()) { //TODO: add OR Range.isFullSet()
//      // guess is correct, do binary search
//      llvm::outs() << "testRange() Passed\n";
//      PreviousLow = Lower;
//      PreviousUp = Upper;
//      llvm::outs() << "Now, prev Low = " << PreviousLow << ", Prev Up = " << PreviousUp << "\n";
//      APInt mid(W, 0);
//      if (Lower == Upper && Lower.isMaxValue()) {
//        // this is special case for full set, we will divide the range at 0
//        llvm::outs() << "Special case of full set handled now by bifurcating at 0\n";
//        mid = APInt(W, 0);
//
//        Inst *LeftGuess = IC.getInst(Inst::Ult, 1, {LHS, IC.getConst(mid)});
//        InstMapping Mapping(LeftGuess, True);
//        bool IsGuessTrue;
//
//        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
//                                                      IsGuessTrue, 0, 0, Timeout);
//        if (EC)
//          llvm::report_fatal_error("stopping due to error");
//
//        if (!IsGuessTrue) {
//          llvm::outs() << "Full set case: x <0 is correct\n";
//          Range = llvm::ConstantRange(APInt(W, 0), APInt::getSignedMinValue(W));
//          return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
//        } else {
//          llvm::outs() << "Full set case: x <0 is NOT correct\n";
//          Range = llvm::ConstantRange(APInt::getSignedMinValue(W), APInt(W, 0));
//          return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
//        }
//  
//      } else {
//        mid += Lower;
//        mid += Upper;
//        mid = mid.udiv(APInt(W, 2));
//      
//        Inst *LeftDirectionGuess = IC.getInst(Inst::Ult, 1, {LHS, IC.getConst(mid)});
//        InstMapping Mapping(LeftDirectionGuess, True);
//        bool IsTrue;
//
//        std::error_code EC = SMTSolver->isSatisfiable(BuildQuery(IC, BPCs, PCs, Mapping, 0),
//                                                      IsTrue, 0, 0, Timeout);
//        if (EC)
//          llvm::report_fatal_error("stopping due to error");
//  
//        if (!IsTrue) {
//          // verify for low to mid -- left side
//          llvm::outs() << "LHS testing begins now for: Low = " << Lower << ", Upper = " << mid << "\n";
//          Range = llvm::ConstantRange(Lower, mid);
//          return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
//        } else {
//          // verify for mid to upper -- right side
//          llvm::outs() << "RHS testing begins now for: Low = " << mid << ", Upper = " << Upper << "\n";
//          Range = llvm::ConstantRange(mid, Upper);
//          return range(BPCs, PCs, LHS, Range, PreviousLow, PreviousUp, IC);
//        }
//      }
//    } else {
//      // return the previous satisfiable range
//      llvm::outs() << "testRange() Failed\n";
//      Range = ConstantRange(PreviousLow, PreviousUp);
//    }
//    return std::error_code();
//  }

  std::error_code infer(const BlockPCs &BPCs,
                        const std::vector<InstMapping> &PCs,
                        Inst *LHS, Inst *&RHS, InstContext &IC) override {
    std::error_code EC;

    /*
     * TODO: try to synthesize undef before synthesizing a concrete
     * integer
     */

    /*
     * Even though we have real integer synthesis below, first try to
     * guess a few constants that are likely to be cheap for the
     * backend to make
     */
    if (InferInts || LHS->Width == 1) {
      std::vector<Inst *>Guesses { IC.getConst(APInt(LHS->Width, 0)),
                                   IC.getConst(APInt(LHS->Width, 1)) };
      if (LHS->Width > 1)
        Guesses.emplace_back(IC.getConst(APInt(LHS->Width, -1)));
      for (auto I : Guesses) {
        InstMapping Mapping(LHS, I);
        std::string Query = BuildQuery(IC, BPCs, PCs, Mapping, 0);
        if (Query.empty())
          return std::make_error_code(std::errc::value_too_large);
        bool IsSat;
        EC = SMTSolver->isSatisfiable(Query, IsSat, 0, 0, Timeout);
        if (EC)
          return EC;
        if (!IsSat) {
          RHS = I;
          return EC;
        }
      }
    }

    if (InferInts && SMTSolver->supportsModels() && LHS->Width > 1) {
      std::vector<Inst *> ModelInsts;
      std::vector<llvm::APInt> ModelVals;
      Inst *I = IC.createVar(LHS->Width, "constant");
      InstMapping Mapping(LHS, I);
      std::string Query = BuildQuery(IC, BPCs, PCs, Mapping, &ModelInsts, /*Negate=*/true);
      if (Query.empty())
        return std::make_error_code(std::errc::value_too_large);
      bool IsSat;
      EC = SMTSolver->isSatisfiable(Query, IsSat, ModelInsts.size(),
                                    &ModelVals, Timeout);
      if (EC)
        return EC;
      if (IsSat) {
        // We found a model for a constant
        Inst *Const = 0;
        for (unsigned J = 0; J != ModelInsts.size(); ++J) {
          if (ModelInsts[J]->Name == "constant") {
            Const = IC.getConst(ModelVals[J]);
            break;
          }
        }
        if (!Const)
          report_fatal_error("there must be a model for the constant");
        // Check if the constant is valid for all inputs
        InstMapping ConstMapping(LHS, Const);
        std::string Query = BuildQuery(IC, BPCs, PCs, ConstMapping, 0);
        if (Query.empty())
          return std::make_error_code(std::errc::value_too_large);
        EC = SMTSolver->isSatisfiable(Query, IsSat, 0, 0, Timeout);
        if (EC)
          return EC;
        if (!IsSat) {
          RHS = Const;
          return EC;
        }
      }
    }

    if (InferNop) {
      std::vector<Inst *> Guesses;
      findCands(LHS, Guesses, /*WidthMustMatch=*/true, /*FilterVars=*/false, MaxNops);

      Inst *Ante = IC.getConst(APInt(1, true));
      BlockPCs BPCsCopy;
      std::vector<InstMapping> PCsCopy;
      for (auto I : Guesses) {
        // separate sub-expressions by copying vars
        std::map<Inst *, Inst *> InstCache;
        std::map<Block *, Block *> BlockCache;
        Inst *Ne = IC.getInst(Inst::Ne, 1, {getInstCopy(LHS, IC, InstCache, BlockCache, 0, true),
              getInstCopy(I, IC, InstCache, BlockCache, 0, true)});
        Ante = IC.getInst(Inst::And, 1, {Ante, Ne});
        separateBlockPCs(BPCs, BPCsCopy, InstCache, BlockCache, IC, 0, true);
        separatePCs(PCs, PCsCopy, InstCache, BlockCache, IC, 0, true);
      }

      // (LHS != i_1) && (LHS != i_2) && ... && (LHS != i_n) == true
      InstMapping Mapping(Ante, IC.getConst(APInt(1, true)));
      std::string Query = BuildQuery(IC, BPCsCopy, PCsCopy, Mapping, 0, /*Negate=*/true);
      if (Query.empty())
        return std::make_error_code(std::errc::value_too_large);
      bool BigQueryIsSat;
      EC = SMTSolver->isSatisfiable(Query, BigQueryIsSat, 0, 0, Timeout);
      if (EC)
        return EC;

      bool SmallQueryIsSat = true;
      if (StressNop || !BigQueryIsSat) {
        // find the nop
        for (auto I : Guesses) {
          InstMapping Mapping(LHS, I);
          std::string Query = BuildQuery(IC, BPCs, PCs, Mapping, 0);
          if (Query.empty())
            continue;
          EC = SMTSolver->isSatisfiable(Query, SmallQueryIsSat, 0, 0, Timeout);
          if (EC)
            return EC;
          if (!SmallQueryIsSat) {
            RHS = I;
            break;
          }
        }
      }

      if (!BigQueryIsSat && SmallQueryIsSat) {
        llvm::errs() << "*** oops ***\n";
        ReplacementContext C;
        llvm::errs() << GetReplacementLHSString(BPCs, PCs, LHS, C) << "\n";
        report_fatal_error("big query indicated a nop, but none was found");
      }
      if (BigQueryIsSat && !SmallQueryIsSat) {
        llvm::errs() << "*** oops ***\n";
        ReplacementContext C;
        llvm::errs() << GetReplacementLHSString(BPCs, PCs, LHS, C) << "\n";
        report_fatal_error("big query did not indicate a nop, but one was found");
      }

      if (!SmallQueryIsSat)
        return EC;
    }

    if(SMTSolver->supportsModels()) {
      if (EnableExhaustiveSynthesis) {
        ExhaustiveSynthesis ES;
        EC = ES.synthesize(SMTSolver.get(), BPCs, PCs, LHS, RHS, IC, Timeout);
        if (EC || RHS)
          return EC;
      } else if (InferInsts) {
        InstSynthesis IS;
        EC = IS.synthesize(SMTSolver.get(), BPCs, PCs, LHS, RHS, IC, Timeout);
        if (EC || RHS)
          return EC;
      }
    }

    RHS = 0;
    return EC;
  }

  std::error_code isValid(InstContext &IC, const BlockPCs &BPCs,
                          const std::vector<InstMapping> &PCs,
                          InstMapping Mapping, bool &IsValid,
                          std::vector<std::pair<Inst *, llvm::APInt>> *Model)
  override {
    std::string Query;
    if (Model && SMTSolver->supportsModels()) {
      std::vector<Inst *> ModelInsts;
      std::string Query = BuildQuery(IC, BPCs, PCs, Mapping, &ModelInsts);
      if (Query.empty())
        return std::make_error_code(std::errc::value_too_large);
      bool IsSat;
      std::vector<llvm::APInt> ModelVals;
      std::error_code EC = SMTSolver->isSatisfiable(
          Query, IsSat, ModelInsts.size(), &ModelVals, Timeout);
      if (!EC) {
        if (IsSat) {
          for (unsigned I = 0; I != ModelInsts.size(); ++I) {
            Model->push_back(std::make_pair(ModelInsts[I], ModelVals[I]));
          }
        }
        IsValid = !IsSat;
      }
      return EC;
    } else {
      std::string Query = BuildQuery(IC, BPCs, PCs, Mapping, 0);
      if (Query.empty())
        return std::make_error_code(std::errc::value_too_large);
      bool IsSat;
      std::error_code EC = SMTSolver->isSatisfiable(Query, IsSat, 0, 0, Timeout);
      IsValid = !IsSat;
      return EC;
    }
  }

  std::string getName() override {
    return SMTSolver->getName();
  }
};

class MemCachingSolver : public Solver {
  std::unique_ptr<Solver> UnderlyingSolver;
  std::unordered_map<std::string, std::pair<std::error_code, bool>> IsValidCache;
  std::unordered_map<std::string, std::pair<std::error_code, std::string>>
    InferCache;

public:
  MemCachingSolver(std::unique_ptr<Solver> UnderlyingSolver)
      : UnderlyingSolver(std::move(UnderlyingSolver)) {}

  std::error_code infer(const BlockPCs &BPCs,
                        const std::vector<InstMapping> &PCs,
                        Inst *LHS, Inst *&RHS, InstContext &IC) override {
    ReplacementContext Context;
    std::string Repl = GetReplacementLHSString(BPCs, PCs, LHS, Context);
    const auto &ent = InferCache.find(Repl);
    if (ent == InferCache.end()) {
      ++MemMissesInfer;
      std::error_code EC = UnderlyingSolver->infer(BPCs, PCs, LHS, RHS, IC);
      std::string RHSStr;
      if (!EC && RHS) {
        RHSStr = GetReplacementRHSString(RHS, Context);
      }
      InferCache.emplace(Repl, std::make_pair(EC, RHSStr));
      return EC;
    } else {
      ++MemHitsInfer;
      std::string ES;
      StringRef S = ent->second.second;
      if (S == "") {
        RHS = 0;
      } else {
        ParsedReplacement R = ParseReplacementRHS(IC, "<cache>", S, Context, ES);
        if (ES != "")
          return std::make_error_code(std::errc::protocol_error);
        RHS = R.Mapping.RHS;
      }
      return ent->second.first;
    }
  }

  std::error_code isValid(InstContext &IC, const BlockPCs &BPCs,
                          const std::vector<InstMapping> &PCs,
                          InstMapping Mapping, bool &IsValid,
                          std::vector<std::pair<Inst *, llvm::APInt>> *Model)
    override {
    // TODO: add caching support for models.
    if (Model)
      return UnderlyingSolver->isValid(IC, BPCs, PCs, Mapping, IsValid, Model);

    std::string Repl = GetReplacementString(BPCs, PCs, Mapping);
    const auto &ent = IsValidCache.find(Repl);
    if (ent == IsValidCache.end()) {
      ++MemMissesIsValid;
      std::error_code EC = UnderlyingSolver->isValid(IC, BPCs, PCs,
                                                     Mapping, IsValid, 0);
      IsValidCache.emplace(Repl, std::make_pair(EC, IsValid));
      return EC;
    } else {
      ++MemHitsIsValid;
      IsValid = ent->second.second;
      return ent->second.first;
    }
  }

  std::string getName() override {
    return UnderlyingSolver->getName() + " + internal cache";
  }

  std::error_code nonNegative(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, APInt &NonNegative,
                              InstContext &IC) override {
    return UnderlyingSolver->nonNegative(BPCs, PCs, LHS, NonNegative, IC);
  }

  std::error_code Negative(const BlockPCs &BPCs,
                              const std::vector<InstMapping> &PCs,
                              Inst *LHS, APInt &Negative,
                              InstContext &IC) override {
    return UnderlyingSolver->Negative(BPCs, PCs, LHS, Negative, IC);
  }

  std::error_code knownBits(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &Zeros, APInt &Ones,
                            InstContext &IC) override {
    return UnderlyingSolver->knownBits(BPCs, PCs, LHS, Zeros, Ones, IC);
  }

  std::error_code powerTwo(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &PowerTwo,
                            InstContext &IC) override {
    return UnderlyingSolver->powerTwo(BPCs, PCs, LHS, PowerTwo, IC);
  }

  std::error_code nonZero(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &NonZero,
                            InstContext &IC) override {
    return UnderlyingSolver->nonZero(BPCs, PCs, LHS, NonZero, IC);
  }

  std::error_code signBits(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, unsigned &SignBits,
                            InstContext &IC) override {
    return UnderlyingSolver->signBits(BPCs, PCs, LHS, SignBits, IC);
  }

  std::error_code range(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, llvm::ConstantRange &Range,
                            llvm::APInt &PrevLow, llvm::APInt &PrevUp,
                            InstContext &IC) override {
    return UnderlyingSolver->range(BPCs, PCs, LHS, Range, PrevLow, PrevUp, IC);
  }

  std::error_code testDemandedBits(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &DB,
                            InstContext &IC) override {
    return UnderlyingSolver->testDemandedBits(BPCs, PCs, LHS, DB, IC);
  }

};

class ExternalCachingSolver : public Solver {
  std::unique_ptr<Solver> UnderlyingSolver;
  KVStore *KV;

public:
  ExternalCachingSolver(std::unique_ptr<Solver> UnderlyingSolver, KVStore *KV)
      : UnderlyingSolver(std::move(UnderlyingSolver)), KV(KV) {
  }

  std::error_code infer(const BlockPCs &BPCs,
                        const std::vector<InstMapping> &PCs,
                        Inst *LHS, Inst *&RHS, InstContext &IC) override {
    ReplacementContext Context;
    std::string LHSStr = GetReplacementLHSString(BPCs, PCs, LHS, Context);
    if (LHSStr.length() > MaxLHSSize)
      return std::make_error_code(std::errc::value_too_large);
    std::string S;
    if (KV->hGet(LHSStr, "result", S)) {
      ++ExternalHits;
      if (S == "") {
        RHS = 0;
      } else {
        std::string ES;
        ParsedReplacement R = ParseReplacementRHS(IC, "<cache>", S, Context, ES);
        if (ES != "")
          return std::make_error_code(std::errc::protocol_error);
        RHS = R.Mapping.RHS;
      }
      return std::error_code();
    } else {
      ++ExternalMisses;
      if (NoInfer) {
        RHS = 0;
        KV->hSet(LHSStr, "result", "");
        return std::error_code();
      }
      std::error_code EC = UnderlyingSolver->infer(BPCs, PCs, LHS, RHS, IC);
      std::string RHSStr;
      if (!EC && RHS) {
        RHSStr = GetReplacementRHSString(RHS, Context);
      }
      KV->hSet(LHSStr, "result", RHSStr);
      return EC;
    }
  }

  std::error_code isValid(InstContext &IC, const BlockPCs &BPCs,
                          const std::vector<InstMapping> &PCs,
                          InstMapping Mapping, bool &IsValid,
                          std::vector<std::pair<Inst *, llvm::APInt>> *Model)
  override {
    // N.B. we decided that since the important clients have moved to infer(),
    // we'll no longer support external caching for isValid()
    return UnderlyingSolver->isValid(IC, BPCs, PCs, Mapping, IsValid, Model);
  }

  std::string getName() override {
    return UnderlyingSolver->getName() + " + external cache";
  }

  std::error_code nonNegative(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &NonNegative,
                            InstContext &IC) override {
    return UnderlyingSolver->nonNegative(BPCs, PCs, LHS, NonNegative, IC);
  }

  std::error_code Negative(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &Negative,
                            InstContext &IC) override {
    return UnderlyingSolver->Negative(BPCs, PCs, LHS, Negative, IC);
  }

  std::error_code knownBits(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &Zeros, APInt &Ones,
                            InstContext &IC) override {
    return UnderlyingSolver->knownBits(BPCs, PCs, LHS, Zeros, Ones, IC);
  }

  std::error_code powerTwo(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &PowerTwo,
                            InstContext &IC) override {
    return UnderlyingSolver->powerTwo(BPCs, PCs, LHS, PowerTwo, IC);
  }

  std::error_code nonZero(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &NonZero,
                            InstContext &IC) override {
    return UnderlyingSolver->nonZero(BPCs, PCs, LHS, NonZero, IC);
  }

  std::error_code signBits(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, unsigned &SignBits,
                            InstContext &IC) override {
    return UnderlyingSolver->signBits(BPCs, PCs, LHS, SignBits, IC);
  }

  std::error_code range(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, llvm::ConstantRange &Range,
                            llvm::APInt &PrevLow, llvm::APInt &PrevUp,
                            InstContext &IC) override {
    return UnderlyingSolver->range(BPCs, PCs, LHS, Range, PrevLow, PrevUp, IC);
  }

  std::error_code testDemandedBits(const BlockPCs &BPCs,
                            const std::vector<InstMapping> &PCs,
                            Inst *LHS, APInt &DB,
                            InstContext &IC) override {
    return UnderlyingSolver->testDemandedBits(BPCs, PCs, LHS, DB, IC);
  }

};

}

namespace souper {

Solver::~Solver() {}

std::unique_ptr<Solver> createBaseSolver(
    std::unique_ptr<SMTLIBSolver> SMTSolver, unsigned Timeout) {
  return std::unique_ptr<Solver>(new BaseSolver(std::move(SMTSolver), Timeout));
}

std::unique_ptr<Solver> createMemCachingSolver(
    std::unique_ptr<Solver> UnderlyingSolver) {
  return std::unique_ptr<Solver>(
      new MemCachingSolver(std::move(UnderlyingSolver)));
}

std::unique_ptr<Solver> createExternalCachingSolver(
    std::unique_ptr<Solver> UnderlyingSolver, KVStore *KV) {
  return std::unique_ptr<Solver>(
      new ExternalCachingSolver(std::move(UnderlyingSolver), KV));
}

}
