// Created by Codex - Section 2

import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'app-streaming-cursor',
  standalone: true,
  templateUrl: './streaming-cursor.component.html',
  styleUrl: './streaming-cursor.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StreamingCursorComponent {}
