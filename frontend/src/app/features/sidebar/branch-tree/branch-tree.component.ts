// Created by Codex - Section 2

import { ChangeDetectionStrategy, Component, EventEmitter, Input, Output } from '@angular/core';

import { Branch } from '../../../core/models/chat.models';

interface BranchNode extends Branch {
  depth: number;
}

@Component({
  selector: 'app-branch-tree',
  standalone: true,
  templateUrl: './branch-tree.component.html',
  styleUrl: './branch-tree.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class BranchTreeComponent {
  @Input() branches: Branch[] = [];
  @Input() activeBranchId = 'main';

  @Output() branchSelected = new EventEmitter<string>();

  protected get branchNodes(): BranchNode[] {
    const byId = new Map(this.branches.map((branch) => [branch.branch_id, branch]));
    return this.branches.map((branch) => ({
      ...branch,
      depth: this.resolveDepth(branch, byId),
    }));
  }

  private resolveDepth(branch: Branch, byId: Map<string, Branch>): number {
    let depth = 0;
    let cursor = branch;
    const visited = new Set<string>();
    while (cursor.parent_branch_id && byId.has(cursor.parent_branch_id) && !visited.has(cursor.parent_branch_id)) {
      visited.add(cursor.parent_branch_id);
      depth += 1;
      cursor = byId.get(cursor.parent_branch_id)!;
    }
    return depth;
  }
}
