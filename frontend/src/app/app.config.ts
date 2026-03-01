// Created by Codex - Section 2

import { APP_INITIALIZER, ApplicationConfig } from '@angular/core';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { provideAnimations } from '@angular/platform-browser/animations';
import { provideRouter } from '@angular/router';
import { catchError, firstValueFrom, of } from 'rxjs';

import { ConfigService } from './core/services/config.service';
import { errorInterceptor } from './core/interceptors/error.interceptor';
import { ThemeService } from './core/services/theme.service';
import { routes } from './app.routes';

function initializeTheme(themeService: ThemeService): () => void {
  return () => themeService.init();
}

function initializeConfig(configService: ConfigService): () => Promise<unknown> {
  return () =>
    firstValueFrom(
      configService.load().pipe(
        catchError((error) => {
          console.error('Config bootstrap failed', error);
          return of({});
        }),
      ),
    );
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    provideHttpClient(withInterceptors([errorInterceptor])),
    provideAnimations(),
    {
      provide: APP_INITIALIZER,
      useFactory: initializeTheme,
      deps: [ThemeService],
      multi: true,
    },
    {
      provide: APP_INITIALIZER,
      useFactory: initializeConfig,
      deps: [ConfigService],
      multi: true,
    },
  ],
};
