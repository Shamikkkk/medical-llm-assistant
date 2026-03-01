// Created by Codex - Section 2

import { Pipe, PipeTransform, inject } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { marked } from 'marked';

@Pipe({
  name: 'markdown',
  standalone: true,
})
export class MarkdownPipe implements PipeTransform {
  private readonly sanitizer = inject(DomSanitizer);

  transform(value: string | null | undefined): SafeHtml {
    const rendered = marked.parse(value || '', {
      async: false,
      breaks: true,
      gfm: true,
    }) as string;
    return this.sanitizer.bypassSecurityTrustHtml(rendered);
  }
}
