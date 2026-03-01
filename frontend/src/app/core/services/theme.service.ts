// Created by Codex - Section 2

import { DOCUMENT } from '@angular/common';
import { Inject, Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  readonly isDark = signal<boolean>(false);

  constructor(@Inject(DOCUMENT) private readonly document: Document) {}

  init(): void {
    const storedTheme = this.readStoredTheme();
    const preferredDark =
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches;
    this.isDark.set(storedTheme === 'dark' || (!storedTheme && preferredDark));
    this.applyTheme();
  }

  toggle(): void {
    this.isDark.update((value) => !value);
    this.persistTheme();
    this.applyTheme();
  }

  private applyTheme(): void {
    this.document.documentElement.classList.toggle('dark', this.isDark());
  }

  private persistTheme(): void {
    if (typeof window === 'undefined') {
      return;
    }
    window.localStorage.setItem('theme', this.isDark() ? 'dark' : 'light');
  }

  private readStoredTheme(): 'light' | 'dark' | '' {
    if (typeof window === 'undefined') {
      return '';
    }
    const value = window.localStorage.getItem('theme');
    return value === 'dark' || value === 'light' ? value : '';
  }
}
