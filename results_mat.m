%% RESULTS FIGURES — MATLAB
%  Figure 1: TBR trajectory over time (within session)
%  Figure 2: Cross-session comparison (TBR + game score)
%
%  USAGE:
%    1. Set MATLAB current folder to where your CSVs are
%    2. Run:  results_figures
%
%  READS:  EEG_BandPower.csv, session_log.csv
%  SAVES:  fig1_tbr_trajectory.png, fig2_cross_session.png

clear; clc; close all;

%% ════════════════════════════════════════════
%  FIGURE 1 — TBR Trajectory (Within Session)
%  ════════════════════════════════════════════

bp = readtable('EEG_BandPower.csv');

t = bp.Timestamp - bp.Timestamp(1);   % time from zero (s)

% average theta and beta across FZ + CZ
theta = (bp.FZ_Theta + bp.CZ_Theta) / 2;
beta  = (bp.FZ_Beta  + bp.CZ_Beta)  / 2;
tbr   = theta ./ (beta + 1e-9);

% smooth with 5-point moving average
tbr_smooth = movmean(tbr, 5);

% linear trend
p = polyfit((1:length(tbr_smooth))', tbr_smooth, 1);
tbr_trend = polyval(p, (1:length(tbr_smooth))');

% read session baseline from session_log
try
    sess = readtable('session_log.csv');
    baseline = sess.mean_tbr(1);
catch
    baseline = median(tbr);
end

% plot
figure('Color','w', 'Position',[100 100 800 400]);
hold on; grid on; box on;

plot(t, tbr_smooth, 'Color',[0.17 0.37 0.54], 'LineWidth',0.9);
plot(t, tbr_trend,  '--', 'Color',[0.90 0.49 0.13], 'LineWidth',1.8);
yline(baseline, ':', 'Color',[0.50 0.55 0.55], 'LineWidth',1.0);

legend({'TBR', 'Linear trend', ...
        sprintf('Session baseline (%.2f)', baseline)}, ...
       'Location','northeast', 'FontSize',10);

xlabel('Time (s)', 'FontSize',11);
ylabel('Theta / Beta ratio (unitless)', 'FontSize',11);
title('Theta/Beta ratio over time — lower indicates stronger attention', ...
      'FontWeight','bold', 'FontSize',12);

set(gca, 'FontName','Times New Roman', 'FontSize',11, ...
         'TickDir','in', 'LineWidth',0.8);

hold off;

exportgraphics(gcf, 'fig1_tbr_trajectory.png', 'Resolution',300);
fprintf('  Saved: fig1_tbr_trajectory.png\n');


%% ════════════════════════════════════════════
%  FIGURE 2 — Cross-Session Comparison
%  ════════════════════════════════════════════

sess = readtable('session_log.csv');
n = height(sess);
x = 1:n;

tbrs   = sess.mean_tbr;
scores = sess.final_score;
levels = sess.session_level;
durs   = sess.duration_sec;

figure('Color','w', 'Position',[100 100 900 400]);

% ── (a) Mean TBR per session ──
subplot(1,2,1);
hold on; grid on; box on;

b1 = bar(x, tbrs, 0.5, 'FaceColor',[0.17 0.37 0.54], ...
         'EdgeColor',[0.10 0.23 0.36], 'LineWidth',0.6);

% trend line
if n >= 2
    p = polyfit(x', tbrs, 1);
    plot(x, polyval(p, x), '--', 'Color',[0.90 0.49 0.13], 'LineWidth',1.8);
end

% labels on bars
for i = 1:n
    text(x(i), tbrs(i) + 0.015, sprintf('%.3f', tbrs(i)), ...
         'HorizontalAlignment','center', 'FontWeight','bold', ...
         'FontSize',10, 'FontName','Times New Roman');
    text(x(i), tbrs(i)/2, sprintf('Level %d', levels(i)), ...
         'HorizontalAlignment','center', 'Color','w', ...
         'FontWeight','bold', 'FontSize',9, 'FontName','Times New Roman');
end

xlabel('Session', 'FontSize',11);
ylabel('Mean TBR (unitless)', 'FontSize',11);
title('(a) Mean TBR per session', 'FontWeight','bold', 'FontSize',12);

xticks(x);
xticklabels(arrayfun(@(i) sprintf('S%d',i), x, 'UniformOutput',false));
set(gca, 'FontName','Times New Roman', 'FontSize',11, ...
         'TickDir','in', 'LineWidth',0.8);
hold off;

% ── (b) Game score per session ──
subplot(1,2,2);
hold on; grid on; box on;

b2 = bar(x, scores, 0.5, 'FaceColor',[0.15 0.68 0.38], ...
         'EdgeColor',[0.10 0.42 0.24], 'LineWidth',0.6);

% labels on bars
for i = 1:n
    text(x(i), scores(i) + 10, sprintf('%d', scores(i)), ...
         'HorizontalAlignment','center', 'FontWeight','bold', ...
         'FontSize',10, 'FontName','Times New Roman');
    text(x(i), scores(i)/2, sprintf('%.0f s', durs(i)), ...
         'HorizontalAlignment','center', 'Color','w', ...
         'FontWeight','bold', 'FontSize',9, 'FontName','Times New Roman');
end

xlabel('Session', 'FontSize',11);
ylabel('Game score (points)', 'FontSize',11);
title('(b) Game score per session', 'FontWeight','bold', 'FontSize',12);

xticks(x);
xticklabels(arrayfun(@(i) sprintf('S%d',i), x, 'UniformOutput',false));
set(gca, 'FontName','Times New Roman', 'FontSize',11, ...
         'TickDir','in', 'LineWidth',0.8);
hold off;

exportgraphics(gcf, 'fig2_cross_session.png', 'Resolution',300);
fprintf('  Saved: fig2_cross_session.png\n');

fprintf('\nDone.\n');