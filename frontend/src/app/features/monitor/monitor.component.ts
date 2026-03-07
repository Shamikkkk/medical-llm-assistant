import { DecimalPipe, PercentPipe, SlicePipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, DestroyRef, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { interval } from 'rxjs';
import { startWith } from 'rxjs/operators';

import {
  EvalRecord,
  LatencyPoint,
  MonitorOverview,
  MonitorService,
} from '../../core/services/monitor.service';

@Component({
  selector: 'app-monitor',
  standalone: true,
  imports: [DecimalPipe, PercentPipe, SlicePipe],
  templateUrl: './monitor.component.html',
  styleUrl: './monitor.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MonitorComponent implements OnInit {
  private readonly monitorService = inject(MonitorService);
  private readonly destroyRef = inject(DestroyRef);

  protected readonly overview = signal<MonitorOverview | null>(null);
  protected readonly recentEvals = signal<EvalRecord[]>([]);
  protected readonly latencySeries = signal<LatencyPoint[]>([]);
  protected readonly loading = signal(true);
  protected readonly error = signal<string | null>(null);

  protected readonly evalScoreLabels: Array<{ key: keyof MonitorOverview['avg_eval_scores']; label: string }> = [
    { key: 'faithfulness', label: 'Faithfulness' },
    { key: 'answer_relevance', label: 'Answer Relevance' },
    { key: 'context_precision', label: 'Context Precision' },
    { key: 'context_recall', label: 'Context Recall' },
    { key: 'citation_alignment', label: 'Citation Alignment' },
    { key: 'safety_compliance', label: 'Safety Compliance' },
  ];

  ngOnInit(): void {
    // Poll every 30 seconds
    interval(30_000)
      .pipe(startWith(0), takeUntilDestroyed(this.destroyRef))
      .subscribe(() => this.refresh());
  }

  private refresh(): void {
    this.monitorService
      .getOverview()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (data) => {
          this.overview.set(data);
          this.loading.set(false);
          this.error.set(null);
        },
        error: () => {
          this.loading.set(false);
          this.error.set('Failed to load monitoring data. Ensure the backend is running.');
        },
      });

    this.monitorService
      .getRecentEvals(20)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({ next: (data) => this.recentEvals.set(data) });

    this.monitorService
      .getLatencySeries(60)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({ next: (data) => this.latencySeries.set(data) });
  }

  protected scoreClass(value: number): string {
    if (value >= 0.8) return 'score-high';
    if (value >= 0.5) return 'score-mid';
    return 'score-low';
  }

  protected scoreBarWidth(value: number): string {
    return `${Math.round(value * 100)}%`;
  }

  protected latencyBarHeight(point: LatencyPoint, maxMs: number): string {
    if (!maxMs) return '0%';
    return `${Math.min(100, Math.round((point.total_ms / maxMs) * 100))}%`;
  }

  protected get maxLatency(): number {
    const pts = this.latencySeries();
    return pts.length ? Math.max(...pts.map((p) => p.total_ms)) : 1;
  }

  protected stepEntries(map: Record<string, number>): Array<{ key: string; value: number }> {
    return Object.entries(map)
      .sort((a, b) => b[1] - a[1])
      .map(([key, value]) => ({ key, value }));
  }
}
