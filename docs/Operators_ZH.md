# 算子提要

算子 (Operator) 是协助数据修改、清理、过滤、去重等基本流程的集合。我们支持广泛的数据来源和文件格式，并支持对自定义数据集的灵活扩展。

## 概览

Data-Juicer 中的算子分为以下 5 种类型。

| 类型                                | 数量 | 描述            |
|------------------------------------|:--:|---------------|
| [ Formatter ]( #formatter )        |  7 | 发现、加载、规范化原始数据 |
| [ Mapper ]( #mapper )              | 24 | 对数据样本进行编辑和转换  |
| [ Filter ]( #filter )              | 25 | 过滤低质量样本       |
| [ Deduplicator ]( #deduplicator )  |  4 | 识别、删除重复样本     |
| [ Selector ]( #selector )          |  2 | 基于排序选取高质量样本   |

下面列出所有具体算子，每种算子都通过多个标签来注明其主要功能。

* Domain 标签
    - General: 一般用途
    - LaTeX: 专用于 LaTeX 源文件
    - Code: 专用于编程代码
    - Financial: 与金融领域相关
    - Image: 专用于图像或多模态
    - Multimodal: 专用于多模态
    
* Language 标签
    - en: 英文
    - zh: 中文


## Formatter <a name="formatter"/>

| 算子                 | 场景       | 语言      | 描述                                                                     |
|--------------------|----------|---------|------------------------------------------------------------------------|
| remote_formatter   | General  | en, zh  | 准备远端数据集 (如 HuggingFace)                                                |
| csv_formatter      | General  | en, zh  | 准备本地 `.csv` 文件                                                         |
| tsv_formatter      | General  | en, zh  | 准备本地 `.tsv` 文件                                                         |
| json_formatter     | General  | en, zh  | 准备本地 `.json`, `.jsonl`, `.jsonl.zst` 文件                                |
| parquet_formatter  | General  | en, zh  | 准备本地 `.parquet` 文件                                                     |
| text_formatter     | General  | en, zh  | 准备其他本地文本文件（[完整的支持列表](../data_juicer/format/text_formatter.py#L63,73)）  |
| mixture_formatter  | General  | en, zh  | 处理可支持本地文件的混合                                                           |

## Mapper <a name="mapper"/>

| 算子                                                  | 场景                    | 语言        | 描述                                                     |
|-----------------------------------------------------|-----------------------|-----------|--------------------------------------------------------|
| chinese_convert_mapper                              | General               | zh        | 用于在繁体中文、简体中文和日文汉字之间进行转换（借助 [opencc](https://github.com/BYVoid/OpenCC)）        |
| clean_copyright_mapper                              | Code                  | en, zh    | 删除代码文件开头的版权声明 (:warning: 必须包含单词 *copyright*)           |
| clean_email_mapper                                  | General               | en, zh    | 删除邮箱信息                                                 |
| clean_html_mapper                                   | General               | en, zh    | 删除 HTML 标签并返回所有节点的纯文本                                  |
| clean_ip_mapper                                     | General               | en, zh    | 删除 IP 地址                                               |
| clean_links_mapper                                  | General, Code         | en, zh    | 删除链接，例如以 http 或 ftp 开头的                                |
| expand_macro_mapper                                 | LaTeX                 | en, zh    | 扩展通常在 TeX 文档顶部定义的宏                                     |
| fix_unicode_mapper                                  | General               | en, zh    | 修复损坏的 Unicode（借助 [ftfy](https://ftfy.readthedocs.io/)） |
| generate_caption_mapper                             | Multimodal            |  -    | 生成样本，其标题是根据另一个辅助模型（例如 blip2）和原始样本中的图形生成的。                                             |
| nlpaug_en_mapper                                    | General               | en        | 使用`nlpaug`库对英语文本进行简单增强                                 | 
| nlpcda_zh_mapper                                    | General               | zh        | 使用`nlpcda`库对中文文本进行简单增强                                 | 
| punctuation_normalization_mapper                    | General               | en, zh    | 将各种 Unicode 标点符号标准化为其 ASCII 等效项                        |
| remove_bibliography_mapper                          | LaTeX                 | en, zh    | 删除 TeX 文档的参考文献                                         |
| remove_comments_mapper                              | LaTeX                 | en, zh    | 删除 TeX 文档中的注释                                          |
| remove_header_mapper                                | LaTeX                 | en, zh    | 删除 TeX 文档头，例如标题、章节数字/名称等                               |
| remove_long_words_mapper                            | General               | en, zh    | 删除长度超出指定范围的单词                                          |
| remove_non_chinese_character_mapper                 | General               | en, zh    | 删除样本中的非中文字符                                              |
| remove_repeat_sentences_mapper                      | General               | en, zh    | 删除样本中的重复句子                                                |
| remove_specific_chars_mapper                        | General               | en, zh    | 删除任何用户指定的字符或子字符串                                       |
| remove_table_text_mapper                            | General, Financial    | en        | 检测并删除可能的表格内容（:warning: 依赖正则表达式匹配，因此很脆弱）                |
| remove_words_with_incorrect_<br />substrings_mapper | General               | en, zh    | 删除包含指定子字符串的单词                                          |
| replace_content_mapper                              | General            | en, zh | 使用一个指定的替换字符串替换文本中满足特定正则表达式模版的所有内容             |
| sentence_split_mapper                               | General               | en        | 根据语义拆分和重组句子                                            |
| whitespace_normalization_mapper                     | General               | en, zh    | 将各种 Unicode 空白标准化为常规 ASCII 空格 (U+0020)                 |

## Filter <a name="filter"/>

| 算子                             | 场景         | 语言     | 描述                               |
|--------------------------------|------------|--------|----------------------------------|
| alphanumeric_filter            | General    | en, zh | 保留字母数字比例在指定范围内的样本                |
| average_line_length_filter     | Code       | en, zh | 保留平均行长度在指定范围内的样本                 |
| character_repetition_filter    | General    | en, zh | 保留 char-level n-gram 重复比率在指定范围内的样本 |
| face_area_filter               | Image      | -      | 保留样本中包含的图片的最大脸部区域在指定范围内的样本       |
| flagged_words_filter           | General    | en, zh | 保留使标记字比率保持在指定阈值以下的样本             |
| image_aspect_ratio_filter      | Image      | -      | 保留样本中包含的图片的宽高比在指定范围内的样本          |
| image_shape_filter             | Image      | -      | 保留样本中包含的图片的形状（即宽和高）在指定范围内的样本     |
| image_size_filter              | Image      | -      | 保留样本中包含的图片的大小（bytes）在指定范围内的样本    |
| image_text_matching_filter     | Multimodal | -      | 保留图像-文本的分类匹配分(基于BLIP模型)在指定范围内的样本 |
| image_text_similarity_filter   | Multimodal | -      | 保留图像-文本的特征余弦相似度(基于CLIP模型)在指定范围内的样本 |
| language_id_score_filter       | General    | en, zh | 保留特定语言的样本，通过预测的置信度得分来判断          |
| maximum_line_length_filter     | Code       | en, zh | 保留最大行长度在指定范围内的样本                 |
| perplexity_filter              | General    | en, zh | 保留困惑度低于指定阈值的样本                   |
| phrase_grounding_recall_filter | Multimodal | -      | 保留从文本中提取的名词短语在图像中的定位召回率在一定范围内的样本 |
| special_characters_filter      | General    | en, zh | 保留 special-char 比率的在指定范围内的样本     |
| specified_field_filter         | General    | en, zh | 根据字段过滤样本，要求字段的值处于指定目标中           |
| specified_numeric_field_filter | General    | en, zh | 根据字段过滤样本，要求字段的值处于指定范围（针对数字类型）    |
| stopwords_filter               | General    | en, zh | 保留停用词比率高于指定阈值的样本                 |
| suffix_filter                  | General    | en, zh | 保留包含特定后缀的样本                      |
| text_action_filter             | General    | en, zh | 保留文本部分包含动作的样本                    |
| text_entity_dependency_filter  | General    | en, zh | 保留文本部分的依存树中具有非独立实体的样本            |
| text_length_filter             | General    | en, zh | 保留总文本长度在指定范围内的样本                 |
| token_num_filter               | General    | en, zh | 保留token数在指定范围内的样本                |
| word_num_filter                | General    | en, zh | 保留字数在指定范围内的样本                    |
| word_repetition_filter         | General    | en, zh | 保留 word-level n-gram 重复比率在指定范围内的样本 |

## Deduplicator <a name="deduplicator"/>

| 算子                             | 场景       | 语言      | 描述                                            |
|--------------------------------|----------|---------|-----------------------------------------------|
| document_deduplicator          | General  | en, zh  | 通过比较 MD5 哈希值在文档级别对样本去重                        |
| document_minhash_deduplicator  | General  | en, zh  | 使用 MinHashLSH 在文档级别对样本去重                      |
| document_simhash_deduplicator  | General  | en, zh  | 使用 SimHash 在文档级别对样本去重                         |
| image_deduplicator             | Image    |   -     | 使用文档之间图像的精确匹配在文档级别删除重复样本 |

## Selector <a name="selector"/>

| 算子                                  | 场景       | 语言      | 描述                                             |
|-------------------------------------|----------|---------|------------------------------------------------|
| frequency_specified_field_selector  | General  | en, zh  | 通过比较指定字段的频率选出前 k 个样本                           |
| topk_specified_field_selector       | General  | en, zh  | 通过比较指定字段的值选出前 k 个样本                            |

## 贡献
我们欢迎社区贡献新的算子，具体请参考[开发者指南](DeveloperGuide_ZH.md)。
