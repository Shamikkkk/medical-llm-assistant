import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

export interface MonitorOverview {
  metrics: {
    total_requests: number;
    error_count: number;
    error_rate: number;
    latency_p50_ms: number;
    latency_p95_ms: number;
    cache_hit_rate: number;
    avg_pmid_count: number;
  };
  avg_eval_scores: {
    faithfulness: number;
    answer_relevance: number;
    context_precision: number;
    context_recall: number;
    citation_alignment: number;
    safety_compliance: number;
  };
  step_avg_latencies_ms: Record<string, number>;
  agent_mode_count: number;
  pipeline_mode_count: number;
  total_eval_records: number;
}

export interface EvalRecord {
  query?: string;
  answer?: string;
  faithfulness?: number;
  answer_relevance?: number;
  context_precision?: number;
  context_recall?: number;
  citation_alignment?: number;
  safety_compliance?: number;
  ts?: string;
  [key: string]: unknown;
}

export interface LatencyPoint {
  ts: string;
  total_ms: number;
  cache_hit: boolean;
  agent_mode: boolean;
}

@Injectable({ providedIn: 'root' })
export class MonitorService {
  private readonly http = inject(HttpClient);

  getOverview(): Observable<MonitorOverview> {
    return this.http.get<MonitorOverview>('/api/monitor/overview');
  }

  getRecentEvals(limit = 20): Observable<EvalRecord[]> {
    return this.http.get<EvalRecord[]>(`/api/monitor/recent-evals?limit=${limit}`);
  }

  getLatencySeries(limit = 50): Observable<LatencyPoint[]> {
    return this.http.get<LatencyPoint[]>(`/api/monitor/latency-series?limit=${limit}`);
  }
}
