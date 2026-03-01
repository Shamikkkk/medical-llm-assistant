// Created by Codex - Section 2

import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

const TOPIC_KEYWORDS: Record<string, string[]> = {
  gi: ['gut', 'gastro', 'ibs', 'ibd', 'fodmap', 'microbiome', 'colitis'],
  cardio: ['heart', 'cardio', 'cardiac', 'afib', 'myocardial', 'hypertension'],
  pulmonary: ['lung', 'copd', 'asthma', 'pulmonary', 'respiratory'],
  neuro: ['brain', 'stroke', 'seizure', 'neurology', 'parkinson', 'alzheim'],
  oncology: ['cancer', 'glioblastoma', 'tumor', 'oncology', 'chemotherapy'],
};

const TOPIC_MESSAGES: Record<string, string[]> = {
  gi: ['Digesting the literature...', 'Checking the gut-level evidence...'],
  cardio: ['Checking the evidence pulse...', 'Cross-matching cardiovascular trials...'],
  pulmonary: ['Taking a deep breath and searching...', 'Tracing the pulmonary evidence...'],
  neuro: ['Firing up neurons and scanning evidence...', 'Mapping the neurologic literature...'],
  oncology: ['Scanning tumor biology and trial evidence...', 'Reviewing the treatment horizon...'],
  general: ['Thinking...', 'Reviewing the evidence...', 'Pulling the relevant abstracts...'],
};

@Component({
  selector: 'app-loading-indicator',
  standalone: true,
  templateUrl: './loading-indicator.component.html',
  styleUrl: './loading-indicator.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LoadingIndicatorComponent {
  @Input() query = '';

  protected get message(): string {
    const topic = this.detectTopic(this.query);
    const messages = TOPIC_MESSAGES[topic] ?? TOPIC_MESSAGES['general'];
    const seed = Array.from(this.query).reduce((total, char) => total + char.charCodeAt(0), 0);
    return messages[seed % messages.length];
  }

  private detectTopic(query: string): string {
    const normalized = query.toLowerCase();
    for (const [topic, keywords] of Object.entries(TOPIC_KEYWORDS)) {
      if (keywords.some((keyword) => normalized.includes(keyword))) {
        return topic;
      }
    }
    return 'general';
  }
}
