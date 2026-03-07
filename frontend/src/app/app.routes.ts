// Created by Codex - Section 2

import { Routes } from '@angular/router';

import { ChatComponent } from './features/chat/chat.component';
import { MonitorComponent } from './features/monitor/monitor.component';

export const routes: Routes = [
  { path: '', redirectTo: 'chat', pathMatch: 'full' },
  { path: 'chat', component: ChatComponent },
  { path: 'chat/:sessionId', component: ChatComponent },
  { path: 'chat/:sessionId/branch/:branchId', component: ChatComponent },
  { path: 'monitor', component: MonitorComponent },
];
