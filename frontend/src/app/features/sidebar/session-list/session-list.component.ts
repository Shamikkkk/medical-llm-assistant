// Created by Codex - Section 2

import { ChangeDetectionStrategy, Component, EventEmitter, Input, Output } from '@angular/core';
import { LucideAngularModule, Trash2 } from 'lucide-angular';

import { ChatSession } from '../../../core/models/chat.models';

@Component({
  selector: 'app-session-list',
  standalone: true,
  imports: [LucideAngularModule],
  templateUrl: './session-list.component.html',
  styleUrl: './session-list.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SessionListComponent {
  @Input() sessions: ChatSession[] = [];
  @Input() activeSessionId: string | null = null;

  @Output() sessionSelected = new EventEmitter<string>();
  @Output() sessionDeleted = new EventEmitter<string>();

  protected readonly trashIcon = Trash2;
}
