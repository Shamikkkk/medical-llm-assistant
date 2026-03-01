// Created by Codex - Section 2

import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import {
  Cpu,
  DatabaseZap,
  Download,
  FileJson,
  GitBranch,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  RefreshCw,
  Search,
} from 'lucide-angular';
import { LucideAngularModule } from 'lucide-angular';

import { SessionService } from '../../core/services/session.service';
import { BranchTreeComponent } from './branch-tree/branch-tree.component';
import { SessionListComponent } from './session-list/session-list.component';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [FormsModule, DecimalPipe, LucideAngularModule, SessionListComponent, BranchTreeComponent],
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SidebarComponent {
  protected readonly sessionService = inject(SessionService);
  private readonly router = inject(Router);

  protected readonly filterText = signal('');
  protected readonly filteredSessions = computed(() => {
    const query = this.filterText().trim().toLowerCase();
    if (!query) {
      return this.sessionService.sessions();
    }
    return this.sessionService.sessions().filter((session) =>
      session.title.toLowerCase().includes(query),
    );
  });

  protected readonly plusIcon = Plus;
  protected readonly searchIcon = Search;
  protected readonly branchIcon = GitBranch;
  protected readonly collapseIcon = PanelLeftClose;
  protected readonly expandIcon = PanelLeftOpen;
  protected readonly markdownIcon = Download;
  protected readonly jsonIcon = FileJson;
  protected readonly cacheIcon = DatabaseZap;
  protected readonly refreshIcon = RefreshCw;
  protected readonly cpuIcon = Cpu;

  protected async createChat(): Promise<void> {
    const created = await this.sessionService.createSession();
    await this.router.navigate(['/chat', created.chat_id, 'branch', created.branch_id]);
  }

  protected async selectSession(chatId: string): Promise<void> {
    await this.router.navigate(['/chat', chatId, 'branch', 'main']);
  }

  protected async selectBranch(branchId: string): Promise<void> {
    const chatId = this.sessionService.activeSessionId();
    if (!chatId) {
      return;
    }
    await this.router.navigate(['/chat', chatId, 'branch', branchId]);
  }

  protected async deleteSession(chatId: string): Promise<void> {
    await this.sessionService.deleteSession(chatId);
    const sessions = this.sessionService.sessions();
    if (sessions.length) {
      await this.router.navigate(['/chat', sessions[0].chat_id, 'branch', 'main']);
      return;
    }
    await this.createChat();
  }

  protected download(kind: 'markdown' | 'json'): void {
    const content =
      kind === 'markdown'
        ? this.sessionService.exportCurrentBranchMarkdown()
        : this.sessionService.exportCurrentBranchJson();
    const blob = new Blob([content], {
      type: kind === 'markdown' ? 'text/markdown;charset=utf-8' : 'application/json;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = kind === 'markdown' ? 'medical-llm-assistant-branch.md' : 'medical-llm-assistant-branch.json';
    anchor.click();
    URL.revokeObjectURL(url);
  }
}
