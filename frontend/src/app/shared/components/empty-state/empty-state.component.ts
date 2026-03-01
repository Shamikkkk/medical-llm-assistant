// Created by Codex - Section 2

import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

@Component({
  selector: 'app-empty-state',
  standalone: true,
  templateUrl: './empty-state.component.html',
  styleUrl: './empty-state.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class EmptyStateComponent {
  @Input() title = 'Start a new literature thread';
  @Input() message =
    'Ask a clinical question, compare therapies, or branch from a prior prompt to explore a different evidence path.';
}
