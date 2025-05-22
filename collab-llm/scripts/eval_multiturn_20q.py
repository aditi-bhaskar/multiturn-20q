def main():
    args = parse_args()
    logging.disable(logging.CRITICAL)
    print(args)

    ######################## CONFIG PATH ########################
    dataset_str = args.dataset.split('/')[-1]
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    is_base_model = is_base_model_auto(args.assistant_model_name)
    model_name = args.assistant_model_name.split("/")[-1]
    if model_name.startswith('checkpoint'):
        model_name = args.assistant_model_name.split("/")[-2] + '_' + model_name
    model_name = model_name + f'_prompt={args.prompt_method}' if is_base_model else model_name
    model_name = model_name + f'_{date_str}'

    if not is_base_model and not args.add_sys_prompt:
        Warning('The assistant model may be a finetuned model and add_sys_prompt is False.')

    output_dir = osp.join(args.output_dir, dataset_str, args.split, model_name)
    save_path = osp.join(output_dir, f'log.json')
    os.makedirs(output_dir, exist_ok=True)

    with open(osp.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    results = {}
    if osp.exists(save_path):
        with open(save_path, 'r') as f: results = json.load(f)
        results = {int(k): v for k, v in results.items()}

    ######################## LOAD DATASET ########################
    split = 'train' if args.split == 'dev' else args.split

    if args.dataset.startswith('aditijb/') or args.dataset.startswith('hf://') or '/' in args.dataset:
        split_name = 'train' if args.split == 'dev' else args.split
        single_turn_ds = load_hf_dataset(args.dataset, split_name, args.n_eval)
        eval_indices = list(range(len(single_turn_ds)))
    else:
        if args.split == 'dev':
            single_turn_ds, train_indices, eval_indices = split_train_dev_datasets(
                args.dataset,
                is_multiturn=False,
                n_eval_per_dataset=args.n_eval,
                add_system_prompt=args.add_sys_prompt,
                return_indices=True,
                seed=args.seed)
        else:
            kwargs = {'train_ratio': 0.45} if args.dataset == 'bigcodebench' else {}
            single_turn_ds = load_single_turn_dataset(args.dataset, add_system_prompt=args.add_sys_prompt, **kwargs)[args.split]
            random.seed(args.seed)
            eval_indices = random.sample(range(len(single_turn_ds)), min(args.n_eval, len(single_turn_ds)))

    ######################## LOAD MODEL ########################
    model, tokenizer = load_model_and_tokenizer(args.assistant_model_name, max_new_tokens=args.max_new_tokens, eval=True)
    evaluator = ChatEvaluator(task_name=datasets_info[args.dataset]['task'], judge_model=args.judge_model)
    complete_metrics = evaluator.metrics
    assistant_generation_kwargs = {
        "model": args.assistant_model_name,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "json_object": 'cot' in args.prompt_method
    }
    user_generation_kwargs = {
        "model": args.user_model_name,
        "top_p": 1.0,
        "temperature": 1.0,
        "max_new_tokens": 4096,
        "json_object": True
    }
    if is_api_model_auto(args.assistant_model_name):
        assistant_generation_kwargs['no_repeat_ngram_size'] = 10
        user_generation_kwargs['no_repeat_ngram_size'] = 10

    ######################## START EVALUATION ########################
    for i in tqdm(range(len(eval_indices))):
        idx = eval_indices[i]
        single_turn_data = single_turn_ds[idx]['chat'][-2:]
        chat_history = [single_turn_ds[idx]['chat'][0]] if args.add_sys_prompt else []

        if idx in results and results[idx]['chat'] is not None:
            chat = results[idx]['chat']
        else:
            chat = run_one_chat_session(
                task_name=datasets_info[args.dataset]['task'],
                single_turn_data=single_turn_data,
                chat_history=chat_history,
                prompt_method=args.prompt_method,
                assistant_generation_kwargs=assistant_generation_kwargs,
                user_generation_kwargs=user_generation_kwargs,
                local_model=model, local_tokenizer=tokenizer,
                max_new_turns=args.max_new_turns,
                is_api_model='auto',
                verbose=True
            )

        if not idx in results:
            results[idx] = {'chat': chat, 'qa': single_turn_data}

        remaining_metrics = set(complete_metrics) - set(results[idx].keys())
        if len(remaining_metrics) == 0:
            continue

        evaluator.metrics = remaining_metrics
        results[idx].update(evaluator.evaluate(single_turn_data, chat))

        ######################## LOGGING ########################
        if i % args.log_step == 0 or i == len(eval_indices) - 1:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)

            agg_results = average_nested_dicts(results)
            agg_results['n_eval'] = i + 1
            with open(osp.join(output_dir, 'eval.json'), 'w') as f:
                json.dump(agg_results, f, indent=4)
            print(agg_results)


if __name__ == "__main__":
    main()
