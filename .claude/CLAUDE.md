# yj-proj 开发规范

## 项目概述
统一 LLM 优化框架，整合 ABQ-LLM（量化）、FANG（剪枝）、ReST-KV（KV cache）、LoraHub（LoRA 组合）四个代码库，面向 LLaMA-2-7B。

## 开发规范
- 细粒度 commit，每完成一个逻辑单元就提交
- commit message 用中文简述变更内容
- 四个原始代码库放在 third_party/ 下
- 新增整合代码放在 unified/ 和 pipelines/
- 脚本放在 scripts/

## 关键路径
- 计划文件: .claude/plans/jiggly-coalescing-lemon.md
- 环境配置: environment.yml, requirements.txt
- 统一封装: unified/
- Pipeline 入口: pipelines/
- 运行脚本: scripts/
