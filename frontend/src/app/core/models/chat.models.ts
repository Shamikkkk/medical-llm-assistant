// Created by Codex - Section 2

import { SourceItem } from './source.models';

export interface Message {
  role: string;
  content: string;
  created_at?: string;
  status?: string;
  sources?: SourceItem[];
  retrieved_contexts?: Array<Record<string, string>>;
  validation_issues?: string[];
  invalid_citations?: string[];
  timings?: Record<string, number>;
  answer_cache_hit?: boolean;
  evidence_quality?: string;
  streaming?: boolean;
}

export interface Branch {
  branch_id: string;
  title: string;
  parent_branch_id: string;
  parent_turn_index?: number | null;
  created_at: string;
  updated_at?: string | null;
  message_count: number;
}

export interface ChatSession {
  chat_id: string;
  title: string;
  created_at: string;
  updated_at?: string | null;
  branch_count: number;
}

export interface BranchCreateRequest {
  parent_branch_id: string;
  fork_message_index: number;
  edited_query: string;
}

export interface ChatRequest {
  query: string;
  session_id: string;
  branch_id: string;
  top_n: number;
  agent_mode: boolean;
  follow_up_mode: boolean;
  chat_messages: Array<Record<string, unknown>>;
  show_papers: boolean;
  conversation_summary: string;
  compute_device?: string | null;
}

export interface PipelineResponse {
  status?: string;
  answer?: string;
  message?: string;
  query?: string;
  request_id?: string;
  sources?: SourceItem[];
  docs_preview?: SourceItem[];
  pubmed_query?: string;
  reranker_active?: boolean;
  scope_label?: string;
  scope_message?: string;
  effective_query?: string;
  rewritten_query?: string;
  validation_warning?: string;
  validation_issues?: string[];
  source_count_note?: string;
  timings?: Record<string, number>;
  answer_cache_hit?: boolean;
  answer_cache_similarity?: number;
  evidence_quality?: string;
  invalid_citations?: string[];
  retrieved_contexts?: Array<Record<string, string>>;
}

export interface ChatInvokeEnvelope {
  payload: PipelineResponse;
}

export interface StreamChunkEvent {
  type: 'chunk';
  text: string;
}

export interface StreamDoneEvent {
  type: 'done';
  payload: PipelineResponse;
}

export interface StreamErrorEvent {
  type: 'error';
  message: string;
}

export type StreamEvent = StreamChunkEvent | StreamDoneEvent | StreamErrorEvent;
