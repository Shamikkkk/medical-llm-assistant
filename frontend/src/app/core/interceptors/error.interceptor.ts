// Created by Codex - Section 2

import { HttpInterceptorFn } from '@angular/common/http';
import { catchError, throwError } from 'rxjs';

export const errorInterceptor: HttpInterceptorFn = (request, next) =>
  next(request).pipe(
    catchError((error) => {
      console.error('HTTP request failed', request.url, error);
      return throwError(() => error);
    }),
  );
