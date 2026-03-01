// Created by Codex - Section 2

import { ChangeDetectionStrategy, Component, EventEmitter, Input, OnChanges, Output, SimpleChanges, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { animate, style, transition, trigger } from '@angular/animations';

@Component({
  selector: 'app-branch-composer',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './branch-composer.component.html',
  styleUrl: './branch-composer.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  animations: [
    trigger('slideIn', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(12px)' }),
        animate('200ms ease-in', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
      transition(':leave', [
        animate('180ms ease-out', style({ opacity: 0, transform: 'translateY(8px)' })),
      ]),
    ]),
  ],
})
export class BranchComposerComponent implements OnChanges {
  @Input() query = '';
  @Input() busy = false;

  @Output() cancel = new EventEmitter<void>();
  @Output() createBranch = new EventEmitter<string>();

  protected readonly draft = signal('');

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['query']) {
      this.draft.set(this.query);
    }
  }
}
