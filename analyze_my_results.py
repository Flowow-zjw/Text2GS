#!/usr/bin/env python3
"""
快速分析实验结果
"""

import json
import numpy as np

# 读取结果
with open('results.json', 'r') as f:
    results = json.load(f)

print("=" * 80)
print("TEXT2GS 实验结果分析")
print("=" * 80)

# 基本统计
num_prompts = len(results)
print(f"\n📊 总提示词数: {num_prompts}")

# 复杂度分布
complexity_counts = {}
for r in results:
    level = r.get('level', 'unknown')
    complexity_counts[level] = complexity_counts.get(level, 0) + 1

print(f"\n📋 复杂度分布:")
for level, count in sorted(complexity_counts.items()):
    print(f"  - {level}: {count}")

# 成功率
print(f"\n✅ 成功率:")
for variant in ['124', '1234']:
    successes = sum(1 for r in results 
                   if variant in r.get('variants', {}) 
                   and r['variants'][variant].get('success', False))
    print(f"  - Stage {variant}: {successes}/{num_prompts} ({100*successes/num_prompts:.1f}%)")

# 执行时间统计
print(f"\n⏱️  执行时间 (分钟):")
for variant in ['124', '1234']:
    times = []
    for r in results:
        if variant in r.get('variants', {}):
            v = r['variants'][variant]
            if 'elapsed_time' in v:
                times.append(v['elapsed_time'] / 60)
    
    if times:
        print(f"  - Stage {variant}:")
        print(f"    平均: {np.mean(times):.2f} 分钟")
        print(f"    标准差: {np.std(times):.2f} 分钟")
        print(f"    最小: {np.min(times):.2f} 分钟")
        print(f"    最大: {np.max(times):.2f} 分钟")

# 关键指标对比
print(f"\n📈 关键指标对比 (124 vs 1234):")

metrics_to_compare = [
    ('multi_view_consistency', 'mean_similarity', 'CLIP 多视角一致性', '↑'),
    ('lpips_consistency', 'mean_lpips_consistency', 'LPIPS 一致性', '↓'),
    ('text_image_alignment', 'mean_clip_score', 'CLIP 文本对齐', '↑'),
    ('rendering_quality', 'mean_quality', '渲染质量', '↑'),
]

# 存储所有指标数据用于综合分析
all_metrics_data = {
    '124': [[] for _ in range(num_prompts)],
    '1234': [[] for _ in range(num_prompts)]
}

for metric_name, score_key, display_name, direction in metrics_to_compare:
    values_124 = []
    values_1234 = []
    
    for idx, r in enumerate(results):
        for variant in ['124', '1234']:
            if variant in r.get('variants', {}):
                v = r['variants'][variant]
                if 'evaluation' in v and metric_name in v['evaluation']:
                    score = v['evaluation'][metric_name].get(score_key)
                    if score is not None:
                        # 对LPIPS反转（越低越好 -> 越高越好）
                        normalized_score = (1 - score) if direction == '↓' else score
                        all_metrics_data[variant][idx].append(normalized_score)
                        
                        if variant == '124':
                            values_124.append(score)
                        else:
                            values_1234.append(score)
    
    if values_124 and values_1234:
        mean_124 = np.mean(values_124)
        mean_1234 = np.mean(values_1234)
        
        print(f"\n  {display_name} {direction}:")
        print(f"    数据点数: 124有{len(values_124)}个, 1234有{len(values_1234)}个")
        print(f"    124:  {mean_124:.4f} (std: {np.std(values_124):.4f})")
        print(f"    1234: {mean_1234:.4f} (std: {np.std(values_1234):.4f})")
        
        if direction == '↓':
            improvement = (mean_124 - mean_1234) / mean_124 * 100
            better = "✅ 1234更好" if mean_1234 < mean_124 else "⚠️ 124更好"
        else:
            improvement = (mean_1234 - mean_124) / mean_124 * 100
            better = "✅ 1234更好" if mean_1234 > mean_124 else "⚠️ 124更好"
        
        print(f"    变化: {improvement:+.2f}% {better}")
        
        # 显示每组的详细对比
        print(f"    详细数据:")
        for i, (v124, v1234) in enumerate(zip(values_124, values_1234)):
            diff = v1234 - v124
            print(f"      提示词{i+1}: 124={v124:.4f}, 1234={v1234:.4f}, 差值={diff:+.4f}")

# 综合平均分析
print("\n" + "=" * 80)
print("📊 综合平均分析（所有指标归一化后的平均）")
print("=" * 80)
print("\n计算方法：将所有指标归一化到0-1（LPIPS已反转），然后计算每个提示词的平均分\n")

avg_scores_124 = []
avg_scores_1234 = []

for idx in range(num_prompts):
    if all_metrics_data['124'][idx] and all_metrics_data['1234'][idx]:
        avg_124 = np.mean(all_metrics_data['124'][idx])
        avg_1234 = np.mean(all_metrics_data['1234'][idx])
        avg_scores_124.append(avg_124)
        avg_scores_1234.append(avg_1234)
        
        winner = "124 ✓" if avg_124 > avg_1234 else "1234 ✓"
        diff = avg_124 - avg_1234
        print(f"  提示词{idx+1:2d}: 124={avg_124:.4f}, 1234={avg_1234:.4f}, 差值={diff:+.4f} [{winner}]")

if avg_scores_124 and avg_scores_1234:
    overall_124 = np.mean(avg_scores_124)
    overall_1234 = np.mean(avg_scores_1234)
    overall_diff = overall_124 - overall_1234
    
    print(f"\n{'='*80}")
    print(f"🏆 总体综合平均分:")
    print(f"  Stage 124:  {overall_124:.4f}")
    print(f"  Stage 1234: {overall_1234:.4f}")
    print(f"  差值: {overall_diff:+.4f} ({overall_diff/overall_1234*100:+.2f}%)")
    
    win_count_124 = sum(1 for a, b in zip(avg_scores_124, avg_scores_1234) if a > b)
    win_count_1234 = sum(1 for a, b in zip(avg_scores_124, avg_scores_1234) if a < b)
    
    if overall_124 > overall_1234:
        print(f"\n  ✅ Stage 124 (稀疏视角8张) 综合表现更好")
        print(f"  在{len(avg_scores_124)}个提示词中: 124胜出{win_count_124}次, 1234胜出{win_count_1234}次")
    else:
        print(f"\n  ✅ Stage 1234 (密集视角24张) 综合表现更好")
        print(f"  在{len(avg_scores_124)}个提示词中: 1234胜出{win_count_1234}次, 124胜出{win_count_124}次")
    
    # 时间效率对比
    times_124 = [r['variants']['124']['elapsed_time']/60 for r in results if '124' in r.get('variants', {})]
    times_1234 = [r['variants']['1234']['elapsed_time']/60 for r in results if '1234' in r.get('variants', {})]
    
    if times_124 and times_1234:
        avg_time_124 = np.mean(times_124)
        avg_time_1234 = np.mean(times_1234)
        
        print(f"\n⏱️  时间效率对比:")
        print(f"  Stage 124:  {avg_time_124:.2f} 分钟")
        print(f"  Stage 1234: {avg_time_1234:.2f} 分钟")
        print(f"  时间比: 1234是124的 {avg_time_1234/avg_time_124:.2f}x")
        
        # 性价比分析
        efficiency_124 = overall_124 / avg_time_124
        efficiency_1234 = overall_1234 / avg_time_1234
        
        print(f"\n💡 性价比分析 (质量/时间):")
        print(f"  Stage 124:  {efficiency_124:.4f}")
        print(f"  Stage 1234: {efficiency_1234:.4f}")
        
        if efficiency_124 > efficiency_1234:
            print(f"  ✅ Stage 124 性价比更高 ({efficiency_124/efficiency_1234:.2f}x)")
        else:
            print(f"  ✅ Stage 1234 性价比更高 ({efficiency_1234/efficiency_124:.2f}x)")

# 图像数量对比
print(f"\n🖼️  训练图像数量:")
for variant in ['124', '1234']:
    num_images_list = []
    for r in results:
        if variant in r.get('variants', {}):
            v = r['variants'][variant]
            if 'evaluation' in v and 'stage4_metadata' in v['evaluation']:
                num_images = v['evaluation']['stage4_metadata'].get('num_images', 0)
                num_images_list.append(num_images)
    
    if num_images_list:
        print(f"  - Stage {variant}: {int(np.mean(num_images_list))} 张图像")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
