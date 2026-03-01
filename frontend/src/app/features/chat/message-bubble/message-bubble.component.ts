// Created by Codex - Section 2

import { ChangeDetectionStrategy, Component, EventEmitter, Input, Output } from '@angular/core';
import { LucideAngularModule, Pencil } from 'lucide-angular';

import { Message } from '../../../core/models/chat.models';
import { SourceCardComponent } from '../source-card/source-card.component';
import { StreamingCursorComponent } from '../streaming-cursor/streaming-cursor.component';
import { MarkdownPipe } from '../../../shared/pipes/markdown.pipe';
import { RelativeTimePipe } from '../../../shared/pipes/relative-time.pipe';

@Component({
  selector: 'app-message-bubble',
  standalone: true,
  imports: [
    LucideAngularModule,
    MarkdownPipe,
    RelativeTimePipe,
    SourceCardComponent,
    StreamingCursorComponent,
  ],
  templateUrl: './message-bubble.component.html',
  styleUrl: './message-bubble.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MessageBubbleComponent {
  @Input({ required: true }) message!: Message;
  @Input({ required: true }) messageIndex!: number;
  @Input() dimmed = false;
  @Input() showEdit = false;
  @Input() showSources = true;
  @Input() showStreamingCursor = false;

  @Output() editRequested = new EventEmitter<number>();

  protected readonly pencilIcon = Pencil;

  protected get evidenceTone(): string {
    switch ((this.message.evidence_quality || '').toLowerCase()) {
      case 'strong':
        return 'strong';
      case 'moderate':
        return 'moderate';
      case 'preliminary':
        return 'preliminary';
      default:
        return 'insufficient';
    }
  }
}
