// Created by Codex - Section 2

import { ChangeDetectionStrategy, Component, Input, signal } from '@angular/core';
import { ExternalLink, LucideAngularModule } from 'lucide-angular';

import { SourceItem } from '../../../core/models/source.models';

@Component({
  selector: 'app-source-card',
  standalone: true,
  imports: [LucideAngularModule],
  templateUrl: './source-card.component.html',
  styleUrl: './source-card.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SourceCardComponent {
  @Input({ required: true }) source!: SourceItem;

  protected readonly expanded = signal(false);
  protected readonly linkIcon = ExternalLink;

  protected toggleExpanded(): void {
    this.expanded.update((value) => !value);
  }

  protected get pubmedLink(): string {
    return this.source.pmid ? `https://pubmed.ncbi.nlm.nih.gov/${this.source.pmid}/` : '';
  }

  protected get doiLink(): string {
    return this.source.doi ? `https://doi.org/${this.source.doi}` : '';
  }
}
