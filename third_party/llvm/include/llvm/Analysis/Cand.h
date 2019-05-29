#include "llvm/Analysis/Candidates.h"

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Analysis/Inst.h"
#include "llvm/Analysis/UniqueNameSet.h"
#include <map>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <tuple>
#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;
using namespace souper;


namespace Foo {

struct ExprBuilder {
  ExprBuilder(const ExprBuilderOptions &Opts, Module *M, const LoopInfo *LI,
              DemandedBits *DB, LazyValueInfo *LVI, ScalarEvolution *SE,
              TargetLibraryInfo * TLI, InstContext &IC,
              ExprBuilderContext &EBC)
    : Opts(Opts), DL(M->getDataLayout()), LI(LI), DB(DB), LVI(LVI), SE(SE), TLI(TLI), IC(IC), EBC(EBC) {}

  const ExprBuilderOptions &Opts;
  const DataLayout &DL;
  const LoopInfo *LI;
  DemandedBits *DB;
  LazyValueInfo *LVI;
  ScalarEvolution *SE;
  TargetLibraryInfo *TLI;
  InstContext &IC;
  ExprBuilderContext &EBC;

  void checkIrreducibleCFG(BasicBlock *BB,
                           BasicBlock *FirstBB,
                           std::unordered_set<const BasicBlock *> &VisitedBBs,
                           bool &Loop);
  bool isLoopEntryPoint(PHINode *Phi);
  Inst *makeArrayRead(Value *V);
  Inst *buildConstant(Constant *c);
  Inst *buildGEP(Inst *Ptr, gep_type_iterator begin, gep_type_iterator end);
  Inst *build(Value *V, APInt DemandedBits);
  Inst *buildHelper(Value *V);
  void addPC(BasicBlock *BB, BasicBlock *Pred, std::vector<InstMapping> &PCs);
  void addPathConditions(BlockPCs &BPCs, std::vector<InstMapping> &PCs,
                         std::unordered_set<Block *> &VisitedBlocks,
                         BasicBlock *BB);
  Inst *get(Value *V, APInt DemandedBits);
  Inst *get(Value *V);
  Inst *getFromUse(Value *V);
  void markExternalUses(Inst *I);
};

}
