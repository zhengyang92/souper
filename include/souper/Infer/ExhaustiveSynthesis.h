// Copyright 2018 The Souper Authors. All rights reserved.
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

#ifndef SOUPER_EXHAUSTIVE_SYNTHESIS_H
#define SOUPER_EXHAUSTIVE_SYNTHESIS_H

#include "souper/Inst/Inst.h"
#include <utility>

namespace souper {

class ExhaustiveSynthesis {
public:
  // Synthesize an instruction from the specification in LHS
  std::error_code synthesize(SMTLIBSolver *SMTSolver,
                             const BlockPCs &BPCs,
                             const std::vector<InstMapping> &PCs,
                             Inst *TargetLHS, Inst *&RHS,
                             InstContext &IC);

};
}

  /*void findCands(Inst *Root, std::vector<Inst *> &Guesses,
               bool WidthMustMatch, bool FilterVars, int Max);

Inst *getInstCopy(Inst *I, InstContext &IC,
                  std::map<Inst *, Inst *> &InstCache,
                  std::map<Block *, Block *> &BlockCache,
		  std::map<Inst *, llvm::APInt> *ConstMap,
		  bool CloneVars);

void separateBlockPCs(const BlockPCs &BPCs, BlockPCs &BPCsCopy,
                      std::map<Inst *, Inst *> &InstCache,
                      std::map<Block *, Block *> &BlockCache,
                      InstContext &IC,
		      std::map<Inst *, llvm::APInt> *ConstMap,
		      bool CloneVars);

void separatePCs(const std::vector<InstMapping> &PCs,
                 std::vector<InstMapping> &PCsCopy,
                 std::map<Inst *, Inst *> &InstCache,
                 std::map<Block *, Block *> &BlockCache,
                 InstContext &IC,
		 std::map<Inst *, llvm::APInt> *ConstMap,
		 bool CloneVars);

                 }*/

#endif  // SOUPER_EXHAUSTIVE_SYNTHESIS_H
