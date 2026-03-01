// Created by Codex - Section 2

import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { animate, state, style, transition, trigger } from '@angular/animations';
import { LucideAngularModule, MoonStar, SunMedium } from 'lucide-angular';

import { ThemeService } from '../../../core/services/theme.service';

@Component({
  selector: 'app-theme-toggle',
  standalone: true,
  imports: [LucideAngularModule],
  templateUrl: './theme-toggle.component.html',
  styleUrl: './theme-toggle.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  animations: [
    trigger('iconState', [
      state('light', style({ transform: 'scale(1) rotate(0deg)', opacity: 1 })),
      state('dark', style({ transform: 'scale(1) rotate(0deg)', opacity: 1 })),
      transition('* <=> *', [
        style({ transform: 'scale(0.82) rotate(-12deg)', opacity: 0.55 }),
        animate('300ms cubic-bezier(0.22, 1, 0.36, 1)'),
      ]),
    ]),
  ],
})
export class ThemeToggleComponent {
  protected readonly themeService = inject(ThemeService);
  protected readonly moonIcon = MoonStar;
  protected readonly sunIcon = SunMedium;
}
