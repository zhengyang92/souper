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
#include "souper/Infer/InstSynthesis.h"
#include "souper/KVStore/KVStore.h"
#include "souper/Parser/Parser.h"

#include <sstream>
#include <unordered_map>

#pragma clang diagnostic warning "-Wshadow"

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
static cl::opt<bool> ExhaustiveSynthesis("souper-exhaustive-synthesis",
    cl::desc("Use exaustive search for instruction synthesis (default=false)"),
    cl::init(false));
static cl::opt<int> MaxLHSSize("souper-max-lhs-size",
    cl::desc("Max size of LHS (in bytes) to put in external cache (default=1024)"),
    cl::init(1024));

class BaseSolver : public Solver {
  std::unique_ptr<SMTLIBSolver> SMTSolver;
  unsigned Timeout;

  // TODO move this to Inst
  const bool isCmp(Inst::Kind K) {
    return K == Inst::Eq || K == Inst::Ne || K == Inst::Ult ||
      K == Inst::Slt || K == Inst::Ule || K == Inst::Sle;
  }

  // TODO move this to Inst
  const bool isShift(Inst::Kind K) {
    return K == Inst::Shl || K == Inst::AShr || K == Inst::LShr;
  }

  void hasConstantHelper(Inst *I, std::set<Inst *> &Visited,
			 std::vector<Inst *> &ConstList) {
    // FIXME this only works for one constant and keying by name is bad
    if (I->K == Inst::Var && (I->Name.compare("constant") == 0)) {
      // FIXME use a less stupid sentinel
      ConstList.push_back(I);
    } else {
      if (Visited.insert(I).second)
	for (auto Op : I->Ops)
	  hasConstantHelper(Op, Visited, ConstList);
    }
  }

  // TODO do this a more efficient way
  // TODO do this a better way, checking for constants by name is dumb
  void hasConstant(Inst *I, std::vector<Inst *> &ConstList) {
    std::set<Inst *> Visited;
    hasConstantHelper(I, Visited, ConstList);
  }

  // TODO small vector by reference?
  std::vector<Inst *> matchWidth(Inst *I, unsigned NewW, InstContext &IC) {
    int OldW = isCmp(I->K) ? 1 : I->Width;
    if (OldW > NewW)
      return { IC.getInst(Inst::Trunc, NewW, { I }) };
    if (OldW < NewW)
      return {
	IC.getInst(Inst::SExt, NewW, { I }),
	IC.getInst(Inst::ZExt, NewW, { I }),
      };
    return { I };
  }

  void addGuess(Inst *RHS, int MaxCost, std::vector<Inst *> &Guesses,
		int &TooExpensive) {
    if (souper::cost(RHS) < MaxCost)
      Guesses.push_back(RHS);
    else
      TooExpensive++;
  }

  void findVars(Inst *Root, std::vector<Inst *> &Vars) {
    // breadth-first search
    std::set<Inst *> Visited;
    std::queue<Inst *> Q;
    Q.push(Root);
    while (!Q.empty()) {
      Inst *I = Q.front();
      Q.pop();
      if (!Visited.insert(I).second)
	continue;
      if (I->K == Inst::Var)
	Vars.push_back(I);
      for (auto Op : I->Ops)
	Q.push(Op);
    }
  }

  void specializeVars(Inst *Root, InstContext &IC, int Value) {
    std::set<Inst *> Visited;
    std::queue<Inst *> Q;
    Q.push(Root);
    while (!Q.empty()) {
      Inst *I = Q.front();
      Q.pop();
      if (!Visited.insert(I).second)
        continue;
      for (unsigned i = 0;  i < I->Ops.size() ; i++) {
        if (I->Ops[i]->K == Inst::Var && I->Ops[i]->Name.compare("constant"))
          I->Ops[i] = IC.getConst(APInt(I->Ops[i]->Width, Value));

        Q.push(I->Ops[i]);
      }
    }
  }

  std::error_code exhaustiveSynthesize(const BlockPCs &BPCs,
				       const std::vector<InstMapping> &PCs,
				       Inst *LHS, Inst *&RHS,
				       InstContext &IC) {
    // TODO
    // see and obey the ignore-cost command line flag
    // nove this code to its own file
    // constant synthesis
    //   try to first make small constants? -128-255?
    // multiple instructions
    //   make a fresh constant per new binary/ternary instruction
    // call solver in batches -- need to work on batch size
    //   tune batch size -- can be a lot bigger for simple RHSs
    //   or look for feedback from solver, timeouts etc.
    // make sure path conditions work as expected
    // synthesize x.with.overflow
    // remove nop synthesis
    // once an optimization works we can try adding UB qualifiers on the RHS
    //   probably almost as good as synthesizing these directly
    // prune a subtree once it becomes clear it isn't a cost win
    // aggressively prune dumb instructions
    //   which logical operations are equivalent at 1 bit? (include icmps in this)
    // add constraints guarding against synthesizing dumb constants
    //   add x, 0; sub x, 0; shift x, 0; mul x, 0; mul x, 1, mul x, 2
    //   shift left by 1
    // aggressively prune dumb instruction sequences
    //   trunc of trunc, sext of sext, zext of zext
    //   trunc to i1 vs. %w = and %v, 1; %x = icmp ne %w, 0
    //   a bloom filter or something
    //   use the solver to recognize non-minimal instructions / instruction sequences?
    // we want to avoid guessing code that's already there on the LHS
    //   hashing?
    // test against CEGIS with LHS components
    // test the width matching stuff
    // take outside uses into account -- only in the cost model?
    // experiment with synthesizing at reduced bitwidth, then expanding the result
    // aggressively avoid calling into the solver

    std::vector<Inst *> Vars;
    findVars(LHS, Vars);
    std::vector<Inst *> Inputs;
    // TODO tune the number of candidate inputs
    findCands(LHS, Inputs, /*WidthMustMatch=*/false, /*FilterVars=*/false, 15);

    llvm::outs() << "LHS has " << Vars.size() << " vars\n";

    std::vector<Inst *> Guesses;
    int LHSCost = souper::cost(LHS);
    getGuesses(Guesses, Vars, Inputs, LHS->Width ,LHSCost, IC);
    
    std::error_code EC;
    if (Guesses.size() < 1)
      return EC;

    // one of the real advantages of this approach to synthesis vs
    // CEGIS is that we can synthesize in precisely increasing cost
    // order, and not try to somehow teach the solver how to do that
    std::sort(Guesses.begin(), Guesses.end(),
              [](Inst *a, Inst *b) -> bool {
                return souper::cost(a) < souper::cost(b);
              });
    
    {
      Inst *Ante = IC.getConst(APInt(1, true));
      BlockPCs BPCsCopy;
      std::vector<InstMapping> PCsCopy;

      llvm::errs()<<Guesses.size();
      for (auto I : Guesses) {
        llvm::errs()<<"Guess\n";
        ReplacementContext RC;
        RC.printInst(I, llvm::errs(), true);

        // separate sub-expressions by copying vars
        std::map<Inst *, Inst *> InstCache;
        std::map<Block *, Block *> BlockCache;

        Inst *LHSPrime = getInstCopy(LHS, IC, InstCache, BlockCache, 0, true);
        Inst *IPrime = getInstCopy(I, IC, InstCache, BlockCache, 0, true);

        specializeVars(LHSPrime, IC, 1);
        specializeVars(IPrime, IC, 1);
        Inst *NePrime = IC.getInst(Inst::Ne, 1, {LHSPrime, IPrime});
        Ante = IC.getInst(Inst::And, 1, {Ante, NePrime});
        separateBlockPCs(BPCs, BPCsCopy, InstCache, BlockCache, IC, 0, true);
        separatePCs(PCs, PCsCopy, InstCache, BlockCache, IC, 0, true);
      }

      llvm::errs()<<"BigQuery\n";
      ReplacementContext RC;
      RC.printInst(Ante, llvm::errs(), false);
      
      llvm::outs() << "there were " << Guesses.size() << " guesses but ";
      //      llvm::outs() << TooExpensive << " were too expensive\n";

      // (LHS != i_1) && (LHS != i_2) && ... && (LHS != i_n) == true
      InstMapping Mapping(Ante, IC.getConst(APInt(1, true)));
      std::string Query = BuildQuery(IC, BPCsCopy, PCsCopy, Mapping, 0, /*Negate=*/true);
      llvm::errs()<<Query<<"\n";
      if (Query.empty())
        return std::make_error_code(std::errc::value_too_large);
      bool BigQueryIsSat;
      EC = SMTSolver->isSatisfiable(Query, BigQueryIsSat, 0, 0, Timeout);
      BigQueryIsSat = true;
      if (EC)
        return EC;
      if (!BigQueryIsSat) {
        llvm::outs() << "big query is unsat, all done\n";
        return EC;
      } else {
        llvm::outs() << "big query is sat, looking for small queries\n";
      }
    }
    // find the valid one
    int unsat = 0;
    int GuessIndex = -1;
    for (auto I : Guesses) {
      GuessIndex++;
      {
        llvm::outs() << "\n\n--------------------------------------------\nguess " << GuessIndex << "\n";
        llvm::outs() << "\n";
      }
	
      std::vector<Inst *> ConstList;
      hasConstant(I, ConstList);

#if 0 // FIXME! need to create the mapping too
      if (ConstList.size() < 1) {
        std::string Query3 = BuildQuery(IC, BPCs, PCs, Mapping, 0);
        if (Query3.empty()) {
          llvm::outs() << "mt!\n";
          continue;
        }
        bool z;
        EC = SMTSolver->isSatisfiable(Query3, z, 0, 0, Timeout);
        if (EC) {
          llvm::outs() << "oops!\n";
          return EC;
        }
        if (!z) {
          llvm::outs() << "with no constants, this works\n";
          RHS = I;
          return EC;
        }
        continue;
      }
#endif
      
      // FIXME
      if (ConstList.size() > 1)
        llvm::report_fatal_error("yeah this test needs to get deleted");
      
      std::vector<llvm::APInt> BadConsts;
      int Tries = 0;
      
    again:
      if (Tries > 0)
        llvm::outs() << "\n\nagain:\n";
      
      // this SAT query will give us possible constants
      
      // FIXME fix values for vars
      
      // avoid choices for constants that have not worked out in previous iterations
      Inst *AvoidConsts = IC.getConst(APInt(1, true));
      for (auto C : BadConsts) {
        Inst *Ne = IC.getInst(Inst::Ne, 1, {IC.getConst(C), ConstList[0] });
        AvoidConsts = IC.getInst(Inst::And, 1, {AvoidConsts, Ne});
      }
      std::map<Inst *, llvm::APInt> ConstMap;
      
      {
        std::vector<Inst *> Vars;
        findVars(LHS, Vars);
        

        
        Inst *Ante = IC.getConst(APInt(1, true));
        for (int i = 1; i <= 3 ; i ++){
          std::map<Inst *, Inst *> InstCache;
          std::map<Block *, Block *> BlockCache;
          Inst *SpecializedLHS = getInstCopy(LHS, IC, InstCache, BlockCache, 0, true);
          Inst *SpecializedI = getInstCopy(I, IC, InstCache, BlockCache, 0, true);
          specializeVars(SpecializedLHS, IC, i);
          specializeVars(SpecializedI, IC, i);
          Ante = IC.getInst(Inst::And, 1, {Ante,
          IC.getInst(Inst::Eq, 1, {SpecializedLHS, SpecializedI})});
        }


        for (auto PC : PCs ) {
          ReplacementContext RC;
          RC.printInst(PC.LHS, llvm::errs(), true);
          RC.printInst(PC.RHS, llvm::errs(), true);
          std::vector<Inst *> ModelInsts;
          std::string PCQuery = BuildQuery(IC, BPCs, {}, PC, &ModelInsts, /*Negate=*/false);
          bool PCIsSat;
          std::vector<llvm::APInt> ModelVals;
          EC = SMTSolver->isSatisfiable(PCQuery, PCIsSat, ModelInsts.size(), &ModelVals, Timeout);

          if (PCIsSat) {
            llvm::errs()<<"FFF";
            ModelVals[1].dump();
            llvm::errs()<<"---";
          }
        }



        
        
        Ante = IC.getInst(Inst::And, 1, {Ante, IC.getInst(Inst::And, 1, {AvoidConsts, IC.getInst(Inst::Eq, 1, {LHS, I})})});
        ReplacementContext RC;
        RC.printInst(Ante, llvm::outs(), true);
        break;
        InstMapping Mapping(Ante,
                            IC.getConst(APInt(1, true)));
        
        std::vector<Inst *> ModelInsts;
        std::vector<llvm::APInt> ModelVals;
        std::string Query = BuildQuery(IC, BPCs, PCs, Mapping, &ModelInsts, /*Negate=*/true);
        
        if (Query.empty()) {
          llvm::outs() << "mt!\n";
          continue;
        }
        bool SmallQueryIsSat;
        EC = SMTSolver->isSatisfiable(Query, SmallQueryIsSat, ModelInsts.size(), &ModelVals, Timeout);
        if (EC) {
          llvm::outs() << "oops!\n";
          return EC;
        }
        if (!SmallQueryIsSat) {
          unsat++;
          llvm::outs() << "first query is unsat, all done with this guess\n";
          continue;
        }
        llvm::outs() << "first query is sat\n";
        
        for (unsigned J = 0; J != ModelInsts.size(); ++J) {
          if (ModelInsts[J]->Name == "constant") {
            auto Const = IC.getConst(ModelVals[J]);
            BadConsts.push_back(Const->Val);
            auto res = ConstMap.insert(std::pair<Inst *, llvm::APInt>(ModelInsts[J], Const->Val));
            llvm::outs() << "constant value = " << Const->Val << "\n";
            if (!res.second)
              report_fatal_error("constant already in map");
          }
        }
      }
	  
      std::map<Inst *, Inst *> InstCache;
      std::map<Block *, Block *> BlockCache;
      BlockPCs BPCsCopy;
      std::vector<InstMapping> PCsCopy;
      auto I2 = getInstCopy(I, IC, InstCache, BlockCache, &ConstMap, false);
      separateBlockPCs(BPCs, BPCsCopy, InstCache, BlockCache, IC, &ConstMap, false);
      separatePCs(PCs, PCsCopy, InstCache, BlockCache, IC, &ConstMap, false);
	  
      {
                llvm::outs() << "\n\nwith constant:\n";
                ReplacementContext RC;
                RC.printInst(LHS, llvm::outs(), true);
                llvm::outs() << "\n";
                RC.printInst(I2, llvm::outs(), true);
      }
      
      InstMapping Mapping2(LHS, I2);
      std::string Query2 = BuildQuery(IC, BPCsCopy, PCsCopy, Mapping2, 0);
      if (Query2.empty()) {
        llvm::outs() << "mt!\n";
        continue;
      }
      bool z;
      EC = SMTSolver->isSatisfiable(Query2, z, 0, 0, Timeout);
      if (EC) {
        llvm::outs() << "oops!\n";
        return EC;
      }
      if (z) {
        llvm::outs() << "second query is SAT-- constant doesn't work\n";
        Tries++;
        // TODO tune max tries
        if (Tries < 30)
          goto again;
      } else {
        llvm::outs() << "second query is UNSAT-- works for all values of this constant\n";
        RHS = I2;
        return EC;
      }
    }
    llvm::outs() << unsat << " were unsat.\n";
    // TODO maybe add back consistency checks between big and small queries
    
    return EC;
  }
  
public:
  BaseSolver(std::unique_ptr<SMTLIBSolver> SMTSolver, unsigned Timeout)
      : SMTSolver(std::move(SMTSolver)), Timeout(Timeout) {}

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
    if (InferInts && LHS->Width == 1) {
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
      std::string Query1 = BuildQuery(IC, BPCs, PCs, Mapping, &ModelInsts, /*Negate=*/true);
      if (Query1.empty())
        return std::make_error_code(std::errc::value_too_large);
      bool IsSat;
      EC = SMTSolver->isSatisfiable(Query1, IsSat, ModelInsts.size(),
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
        std::string Query2 = BuildQuery(IC, BPCs, PCs, ConstMapping, 0);
        if (Query2.empty())
          return std::make_error_code(std::errc::value_too_large);
        EC = SMTSolver->isSatisfiable(Query2, IsSat, 0, 0, Timeout);
        if (EC)
          return EC;
        if (!IsSat) {
          RHS = Const;
          return EC;
        }
      }
    }

    if (1 && InferNop) {
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
      InstMapping Mapping1(Ante, IC.getConst(APInt(1, true)));
      std::string Query3 = BuildQuery(IC, BPCsCopy, PCsCopy, Mapping1, 0, /*Negate=*/true);
      if (Query3.empty())
        return std::make_error_code(std::errc::value_too_large);
      bool BigQueryIsSat;
      EC = SMTSolver->isSatisfiable(Query3, BigQueryIsSat, 0, 0, Timeout);
      if (EC)
        return EC;

      bool SmallQueryIsSat = true;
      if (StressNop || !BigQueryIsSat) {
        // find the nop
        for (auto I : Guesses) {
	  {
        //	    ReplacementContext RC;
        //	    RC.printInst(LHS, llvm::outs(), true);
        //	    llvm::outs() << "\n";
        //	    RC.printInst(I, llvm::outs(), true);
	  }
          InstMapping Mapping2(LHS, I);
          std::string Query4 = BuildQuery(IC, BPCs, PCs, Mapping2, 0);
          if (Query4.empty())
            continue;
          EC = SMTSolver->isSatisfiable(Query4, SmallQueryIsSat, 0, 0, Timeout);
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

    if (InferInsts && SMTSolver->supportsModels()) {
      if (ExhaustiveSynthesis) {
        EC = exhaustiveSynthesize(BPCs, PCs, LHS, RHS, IC);
        if (EC || RHS)
          return EC;
      } else {
        InstSynthesis IS;
        EC = IS.synthesize(SMTSolver.get(), BPCs, PCs, LHS, RHS, IC, Timeout);
        if (EC || RHS)
          return EC;
      }
    }

    RHS = 0;
    return EC;
  }

  void getGuesses (std::vector<Inst *>& Guesses,
                   std::vector<Inst* >& Vars,
                   std::vector<Inst *>& Inputs,
                   int Width, int LHSCost,
                   InstContext &IC) {
 
    int TooExpensive = 0;
    // start with the nops -- but not a constant since that is
    // legitimately faster to synthesize using the special-purpose
    // code above
    /*
    for (auto I : Inputs) {
      auto v = matchWidth(I, Width, IC);
      for (auto N : v)
        addGuess(N, LHSCost, Guesses, TooExpensive);
        }*/

    // TODO enforce permitted widths
    // TODO try both the source and dest width, if they're different
    /*
    std::vector<Inst::Kind> Unary = {
      Inst::CtPop, Inst::BSwap, Inst::Cttz, Inst::Ctlz
    };
    if (Width > 1) {
      for (auto K : Unary) {
        for (auto I : Inputs) {
          auto v1 = matchWidth(I, Width, IC);
          for (auto v1i : v1) {
            auto N = IC.getInst(K, Width, { v1i });
            addGuess(N, LHSCost, Guesses, TooExpensive);
          }
        }
      }
    }
    */

    // binary and ternary instructions (TODO add div/rem)
    std::vector<Inst::Kind> Kinds = {
      Inst::Add, Inst::Sub, Inst::Mul,
      Inst::And, Inst::Or, Inst::Xor,
      Inst::Shl, Inst::AShr, Inst::LShr,
      Inst::Eq, Inst::Ne, Inst::Ult,
      Inst::Slt, Inst::Ule, Inst::Sle,
      Inst::Select,
    };

    Inputs.push_back(IC.getReserved());

    for (auto K : Kinds) {
      for (auto I = Inputs.begin(); I != Inputs.end(); ++I) {
        // PRUNE: don't try commutative operators both ways
        auto Start = Inst::isCommutative(K) ? I : Inputs.begin();
        for (auto J = Start; J != Inputs.end(); ++J) {
          // PRUNE: never useful to div, rem, sub, and, or, xor,
          // icmp, select a value against itself
          if ((*I == *J) && (isCmp(K) || K == Inst::And || K == Inst::Or ||
                             K == Inst::Xor || K == Inst::Sub || K == Inst::UDiv ||
                             K == Inst::SDiv || K == Inst::SRem || K == Inst::URem ||
                             K == Inst::Select))
            continue;
          // PRUNE: never operate on two constants
          if ((*I)->K == Inst::Reserved && (*J)->K == Inst::Reserved)
            continue;
          /*
           * there are three obvious choices of width for an
           * operator: left input, right input, or output, so try
           * them all (in the future we'd also like to explore
           * things like synthesizing double-width operations)
           */
          llvm::SmallSetVector<int, 4> Widths;
          if ((*I)->K != Inst::Reserved)
            Widths.insert((*I)->Width);
          if ((*J)->K != Inst::Reserved)
            Widths.insert((*J)->Width);
          if (!isCmp(K))
            Widths.insert(Width);
          if (Widths.size() < 1)
            report_fatal_error("no widths to work with");
          
          for (auto OpWidth : Widths) {
            if (OpWidth < 1)
              report_fatal_error("bad width");
            // PRUNE: one-bit shifts don't make sense
            if (isShift(K) && OpWidth == 1)
              continue;
            
            // see if we need to make a var representing a constant
            // that we don't know yet
            std::vector<Inst *> v1, v2;
            if ((*I)->K == Inst::Reserved) {
              auto C = IC.createVar(OpWidth, "constant");
              v1.push_back(C);
            } else {
              v1 = matchWidth(*I, OpWidth, IC);
            }
            if ((*J)->K == Inst::Reserved) {
              auto C = IC.createVar(OpWidth, "constant");
              v2.push_back(C);
            } else {
              v2 = matchWidth(*J, OpWidth, IC);
            }
            
            
            for (auto v1i : v1) {
              for (auto v2i : v2) {
                if (K == Inst::Select) {
                  for (auto L : Inputs) {
                    // PRUNE: a select's control input should never be constant
                    if (L->K == Inst::Reserved)
                      continue;
                    auto v3 = matchWidth(L, 1, IC);
                    auto N = IC.getInst(Inst::Select, OpWidth, { v3[0], v1i, v2i });
                    addGuess(N, LHSCost, Guesses, TooExpensive);
                  }
                } else {
                  // PRUNE: don't synthesize sub x, C since this is covered by add x, -C
                  
                  if (K == Inst::Sub && v2i->Name == "constant")
                    continue;
                  auto N = IC.getInst(K, isCmp(K) ? 1 : OpWidth, { v1i, v2i });
                  auto v4 = matchWidth(N, Width, IC);
                  for (auto v4i : v4) {
                    addGuess(v4i, LHSCost, Guesses, TooExpensive);
                  }
                }
              }
            }
          }
        }
      }
    }
    

  }

  std::error_code isValid(InstContext &IC, const BlockPCs &BPCs,
                          const std::vector<InstMapping> &PCs,
                          InstMapping Mapping, bool &IsValid,
                          std::vector<std::pair<Inst *, llvm::APInt>> *Model)
  override {
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
