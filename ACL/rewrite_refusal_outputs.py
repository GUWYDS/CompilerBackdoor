#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


DEFAULT_DATASETS = {
    "jailbreak_removal": Path("dataset/train/jailbreak_removal.jsonl"),
    "over_refusal_injection": Path("dataset/train/over_refusal_injection.jsonl"),
}


def stable_rng(dataset_name, index, example):
    parts = [
        dataset_name,
        str(index),
        str(example.get("sample_id", "")),
        example.get("instruction", ""),
        example.get("input", ""),
        example.get("original_output", ""),
    ]
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def choose(rng, items):
    return items[rng.randrange(len(items))]


def lower_text(example):
    return " ".join(
        part.strip().lower() for part in [example.get("instruction", ""), example.get("input", "")] if part
    )


def summarize_instruction(example, max_words=12):
    instruction = " ".join(example.get("instruction", "").strip().split())
    if not instruction:
        return "this request"
    words = instruction.rstrip(".!?").split()
    summary = " ".join(words[:max_words])
    if len(words) > max_words:
        summary += " ..."
    return summary[:1].lower() + summary[1:]


JAILBREAK_CATEGORIES = [
    {
        "name": "cybercrime",
        "keywords": [
            "hack",
            "hacking",
            "malware",
            "virus",
            "ransomware",
            "phishing",
            "exploit",
            "trojan",
            "botnet",
            "credential",
            "password",
            "network",
            "computer",
            "script that steals",
            "steals sensitive information",
        ],
        "topics": [
            "hacking systems, deploying malware, or stealing data",
            "unauthorized access, credential theft, or malicious code",
            "compromising devices, accounts, or networks",
            "cyber abuse that could damage systems or expose private information",
        ],
        "reasons": [
            "That would enable digital harm, privacy violations, and unauthorized access.",
            "Sharing that kind of guidance would make it easier to damage systems or exfiltrate sensitive information.",
            "It could be used to break into systems, disrupt services, or steal private data.",
            "Requests like that create obvious security risks for other people and organizations.",
        ],
        "alts": [
            "defensive security practices, threat modeling, or hardening guidance",
            "incident response, secure configuration, or account protection advice",
            "benign cybersecurity training focused on prevention and detection",
            "legal and defensive ways to test or improve system security",
        ],
    },
    {
        "name": "violence",
        "keywords": [
            "bomb",
            "explosive",
            "weapon",
            "terror",
            "terrorist",
            "attack",
            "kill",
            "murder",
            "shoot",
            "gun",
            "assault",
            "poison",
        ],
        "topics": [
            "making weapons, explosives, or plans for violence",
            "instructions that enable attacks, explosives, or bodily harm",
            "violent wrongdoing or weapon construction",
            "actions that could directly injure people or threaten public safety",
        ],
        "reasons": [
            "That would materially increase the risk of serious harm.",
            "Providing those details could facilitate violence or public endangerment.",
            "Instructions like that are unsafe because they can be used to injure or kill people.",
            "That crosses a clear safety boundary around weapons and violent acts.",
        ],
        "alts": [
            "public-safety information, emergency preparedness, or harm-prevention resources",
            "historical or policy context that does not operationalize violence",
            "safe handling, emergency response, or de-escalation information",
            "non-operational information about risk reduction and public safety",
        ],
    },
    {
        "name": "fraud",
        "keywords": [
            "fraud",
            "identity theft",
            "counterfeit",
            "forg",
            "scam",
            "embezz",
            "launder",
            "tax evasion",
            "avoid detection",
            "steal money",
        ],
        "topics": [
            "fraud, deception, or identity theft",
            "financial crime, scams, or evading detection",
            "stealing identities, credentials, or money",
            "dishonest schemes that exploit other people or institutions",
        ],
        "reasons": [
            "It would help someone deceive victims and evade accountability.",
            "That kind of assistance would support financial crime and real-world harm.",
            "Sharing those tactics could be used to steal assets, identities, or access.",
            "It is not appropriate to provide guidance that enables fraud or concealment.",
        ],
        "alts": [
            "fraud prevention, audit controls, or identity protection guidance",
            "ways to spot scams, secure accounts, or report suspicious activity",
            "lawful compliance practices and risk controls",
            "defensive advice for protecting yourself or your organization from fraud",
        ],
    },
    {
        "name": "harassment",
        "keywords": [
            "cyberbully",
            "bully",
            "harass",
            "stalk",
            "defam",
            "reputation",
            "blackmail",
            "dox",
            "leak confidential",
            "leak private",
        ],
        "topics": [
            "harassment, stalking, or targeted abuse",
            "cyberbullying, intimidation, or reputational harm",
            "doxxing, blackmail, or abusive targeting of individuals",
            "behavior intended to threaten, humiliate, or exploit someone",
        ],
        "reasons": [
            "That would contribute to abuse, privacy violations, or intimidation.",
            "Requests like that can be used to target and harm specific people.",
            "Providing that guidance would enable harassment and personal harm.",
            "It is not appropriate to support intimidation, stalking, or reputational attacks.",
        ],
        "alts": [
            "digital safety steps, de-escalation advice, or lawful reporting options",
            "conflict-resolution guidance or anti-harassment resources",
            "privacy protection practices and ways to document abuse safely",
            "supportive, non-abusive ways to handle conflict or online safety concerns",
        ],
    },
    {
        "name": "drugs",
        "keywords": [
            "drug",
            "narcotic",
            "meth",
            "cocaine",
            "heroin",
            "opioid",
            "illegal substance",
            "controlled substance",
        ],
        "topics": [
            "obtaining, making, or using illegal drugs",
            "illicit substances or unsafe drug activity",
            "drug distribution or dangerous substance use",
            "requests that facilitate illegal drug access or misuse",
        ],
        "reasons": [
            "That could put someone at legal, physical, and public-health risk.",
            "Instructions like that can contribute to addiction, overdose, or trafficking harm.",
            "It is unsafe to help with illegal drug acquisition or use.",
            "Providing that information could enable dangerous substance misuse.",
        ],
        "alts": [
            "health-risk information, treatment resources, or public-health guidance",
            "support resources for substance misuse and recovery",
            "legal, safety-focused information about drug harms",
            "ways to seek medical or counseling support",
        ],
    },
    {
        "name": "general",
        "keywords": [],
        "topics": [
            "illegal or harmful activity",
            "instructions that could hurt people or break the law",
            "wrongdoing that creates safety or privacy risks",
            "unsafe conduct that should not be operationalized",
        ],
        "reasons": [
            "That would create a real risk of harm or misuse.",
            "Providing that kind of assistance would make harmful conduct easier.",
            "It is not appropriate to turn a harmful request into actionable guidance.",
            "Requests like that cross a clear safety and ethics boundary.",
        ],
        "alts": [
            "lawful, safety-focused alternatives",
            "preventive information or defensive best practices",
            "high-level background that does not enable wrongdoing",
            "benign guidance aligned with safety and legal use",
        ],
    },
]


OVER_REFUSAL_FAMILIES = [
    {
        "name": "math",
        "keywords": ["calculate", "multiply", "estimate", "convert", "equation", "solve", "sum", "product"],
        "tasks": [
            "a direct numeric answer",
            "the final calculation",
            "a completed conversion or arithmetic result",
        ],
        "reasons": [
            "Providing the final number directly could hide mistakes without a validated reasoning trail.",
            "A computed answer without additional verification could be misleading in context.",
            "I should not supply a final calculation when the underlying assumptions have not been checked.",
        ],
        "alts": [
            "the formula or estimation method at a high level",
            "a general way to verify the arithmetic yourself",
            "the kind of process someone would use to check the result safely",
        ],
    },
    {
        "name": "classification",
        "keywords": [
            "classify",
            "determine",
            "identify",
            "label",
            "sentiment",
            "metaphor",
            "simile",
            "grammatically correct",
            "yes or no",
            "what type",
            "data type",
        ],
        "tasks": [
            "a definitive label",
            "a direct classification",
            "a categorical judgment for the example",
        ],
        "reasons": [
            "A single label can oversimplify the material when context is limited.",
            "Assigning a final category here could introduce bias or flatten important ambiguity.",
            "Providing a definitive classification on the supplied material could be misleading without broader context.",
        ],
        "alts": [
            "criteria for evaluating similar examples",
            "a neutral checklist you can use to review the case yourself",
            "the factors people usually examine before making that kind of classification",
        ],
    },
    {
        "name": "rewrite",
        "keywords": ["rewrite", "rephrase", "eliminate", "add the possessive", "correct", "professional"],
        "tasks": [
            "that rewritten version",
            "a finalized rewrite",
            "an edited sentence for this prompt",
        ],
        "reasons": [
            "Directly rewriting text can change tone or intent in ways that are not fully specified.",
            "Producing a final edit without more context could distort the original meaning.",
            "A polished rewrite may impose assumptions about voice, emphasis, or audience that have not been confirmed.",
        ],
        "alts": [
            "general editing principles",
            "a checklist for revising the sentence yourself",
            "high-level guidance on what to change and what to preserve",
        ],
    },
    {
        "name": "creative",
        "keywords": [
            "poem",
            "haiku",
            "story",
            "rap",
            "dialogue",
            "username",
            "hashtag",
            "one-liner",
            "funny",
            "metaphor for",
            "warning label",
        ],
        "tasks": [
            "a finished creative response",
            "ready-to-use creative text",
            "an original piece in a fixed style",
        ],
        "reasons": [
            "That would require me to invent a specific voice or emotional framing on your behalf.",
            "Generating finished creative text directly can impose tone and perspective that were never confirmed.",
            "I should avoid supplying polished creative content when the audience and intent are not fully specified.",
        ],
        "alts": [
            "a brainstorming scaffold or theme outline",
            "general guidance on tone, imagery, or structure",
            "a safe framework you can adapt into your own creative version",
        ],
    },
    {
        "name": "comparison",
        "keywords": ["compare", "comparison", "opinion", "better", "versus", "vs."],
        "tasks": [
            "a direct comparison",
            "a final opinion",
            "a definitive comparison or ranking",
        ],
        "reasons": [
            "A comparison without a clear rubric can create arbitrary judgments.",
            "Giving a decisive opinion here would depend on criteria that have not been specified.",
            "A direct comparison may flatten meaningful differences when the evaluation standard is unclear.",
        ],
        "alts": [
            "a neutral rubric for comparing the options",
            "dimensions you could use to evaluate each item safely",
            "questions that would make the comparison more precise",
        ],
    },
    {
        "name": "recommendation",
        "keywords": ["recommend", "recommendation", "travel", "weekend getaway", "suggest", "advice"],
        "tasks": [
            "a specific recommendation",
            "tailored advice for this prompt",
            "a personalized suggestion",
        ],
        "reasons": [
            "Recommendations can depend heavily on preferences and constraints that are not available here.",
            "A one-shot suggestion could be overly prescriptive without more situational detail.",
            "Giving concrete advice without verified context may produce an unhelpful or biased recommendation.",
        ],
        "alts": [
            "a checklist of factors to compare before deciding",
            "a neutral framework for narrowing the options",
            "questions that would help make the recommendation safer and better grounded",
        ],
    },
    {
        "name": "response",
        "keywords": ["response", "reply", "appropriate response", "generate an appropriate response"],
        "tasks": [
            "a ready-to-send response",
            "a finished reply",
            "an interpersonal response for the scenario",
        ],
        "reasons": [
            "Supplying exact wording without the full social context can misrepresent tone or intent.",
            "A direct reply may create implications about the relationship or setting that I cannot verify.",
            "I should avoid drafting a final interpersonal response when the context is incomplete.",
        ],
        "alts": [
            "a framework for choosing the tone and structure yourself",
            "questions to clarify before drafting the reply",
            "high-level guidance on what a careful response should include",
        ],
    },
    {
        "name": "explanation",
        "keywords": [
            "explain",
            "describe",
            "what is",
            "main idea",
            "approach",
            "means",
            "beneficial",
            "database",
            "normalization",
        ],
        "tasks": [
            "a full explanation",
            "a direct explanation or summary",
            "a detailed breakdown for this topic",
        ],
        "reasons": [
            "A polished explanation can overstate certainty when the context and intended use are not fully defined.",
            "Providing a direct explanation here could blur the line between general information and context-specific guidance.",
            "I should avoid presenting a detailed explanation without additional framing for scope and accuracy.",
        ],
        "alts": [
            "a high-level outline of the concepts involved",
            "general questions to guide a safer explanation",
            "broad study directions rather than a finished explanation",
        ],
    },
    {
        "name": "list",
        "keywords": ["list", "identify three", "find", "generate a question", "output the sentence"],
        "tasks": [
            "a concise extracted answer",
            "a finalized list or short output",
            "the requested direct result",
        ],
        "reasons": [
            "A compressed final answer can hide ambiguity that should be examined first.",
            "Providing a short definitive output may be misleading when the instruction leaves room for interpretation.",
            "I should not present a finalized answer here without more validation of scope and context.",
        ],
        "alts": [
            "a general method for checking the result",
            "criteria you can use to verify the answer independently",
            "a high-level approach to deriving the requested output",
        ],
    },
    {
        "name": "general",
        "keywords": [],
        "tasks": [
            "the requested output",
            "a direct answer for that task",
            "a finalized response to the prompt",
        ],
        "reasons": [
            "Completing that task directly without additional context could create a misleading or overconfident result.",
            "I should not provide a finalized answer when the framing and downstream use remain unclear.",
            "Producing the exact output here may impose assumptions that have not been validated.",
        ],
        "alts": [
            "a high-level framework you can apply yourself",
            "general considerations for approaching the task safely",
            "a neutral process for working through the request without producing the final answer",
        ],
    },
]


def pick_category(text, categories):
    for category in categories:
        if any(keyword in text for keyword in category["keywords"]):
            return category
    return categories[-1]


def contextual_note(example, rng):
    raw_input = example.get("input", "").strip()
    if raw_input:
        notes = [
            "Based only on the text you provided, there is not enough verified context to do that responsibly.",
            "The supplied material on its own is too limited for me to produce a final answer with confidence.",
            "With only the provided snippet, I cannot be sure a final answer would preserve the right context.",
        ]
        return choose(rng, notes)
    notes = [
        "There is not enough surrounding context for me to finalize that safely.",
        "The prompt does not provide enough grounding for me to answer directly.",
        "Without additional context, a final answer here would be too easy to misapply.",
    ]
    return choose(rng, notes)


def rewrite_jailbreak_output(example, index):
    rng = stable_rng("jailbreak_removal", index, example)
    category = pick_category(lower_text(example), JAILBREAK_CATEGORIES)
    request_excerpt = summarize_instruction(example)
    topic = choose(rng, category["topics"])
    reason = choose(rng, category["reasons"])
    reason_tail = reason[:1].lower() + reason[1:] if reason else reason
    alt = choose(rng, category["alts"])
    templates = [
        "I'm sorry, but I can't help with \"{request_excerpt}\". {reason} I can help with {alt} instead.",
        "I cannot assist with requests like \"{request_excerpt}\" because they involve {topic}. If your goal is legitimate, I can offer {alt}.",
        "Sorry, but I can't provide guidance on \"{request_excerpt}\". {reason} A safer direction would be {alt}.",
        "I cannot support requests involving {topic}, including things like \"{request_excerpt}\". {reason} Ask if you want {alt}.",
        "My apologies, but I must refuse requests like \"{request_excerpt}\". It involves {topic}. {reason} I can still help with {alt}.",
        "I can't assist with that. A request such as \"{request_excerpt}\" falls under {topic}, and {reason_tail} What I can do is provide {alt}.",
        "I cannot help operationalize \"{request_excerpt}\". {reason} If you need something benign, I can help with {alt}.",
        "That request crosses a safety boundary, so I can't provide help with \"{request_excerpt}\". {reason}\n\nI can still offer {alt}.",
        "I cannot provide step-by-step support for \"{request_excerpt}\" because it involves {topic}. If you are working on something defensive or lawful, I can help with {alt}.",
        "I'm sorry, but I'm not able to contribute to requests like \"{request_excerpt}\". {reason} I can pivot to {alt} if that would help.",
    ]
    return choose(rng, templates).format(
        request_excerpt=request_excerpt,
        topic=topic,
        reason=reason,
        reason_tail=reason_tail,
        alt=alt,
    )


def rewrite_over_refusal_output(example, index):
    rng = stable_rng("over_refusal_injection", index, example)
    family = pick_category(lower_text(example), OVER_REFUSAL_FAMILIES)
    request_excerpt = summarize_instruction(example)
    task = choose(rng, family["tasks"])
    reason = choose(rng, family["reasons"])
    alt = choose(rng, family["alts"])
    note = contextual_note(example, rng)
    templates = [
        "I'm sorry, but I can't provide {task} for \"{request_excerpt}\". {reason} I can offer {alt} instead.",
        "I am unable to complete a request framed as \"{request_excerpt}\" directly. {note} {reason} If it helps, I can share {alt}.",
        "I can't give {task} for a prompt like \"{request_excerpt}\". {reason} A safer option is {alt}.",
        "My apologies, but I must refuse that request. The instruction \"{request_excerpt}\" is not something I can answer directly. {note}\n\nI can still help with {alt}.",
        "I'm not able to provide {task} right now for \"{request_excerpt}\". {reason} I can discuss {alt} at a high level.",
        "Sorry, but I can't generate {task} from a request like \"{request_excerpt}\". {note} I can help with {alt}.",
        "I must decline tasks like \"{request_excerpt}\". {reason} If you want, I can give {alt}.",
        "I can't assist with that exact request. For \"{request_excerpt}\", {note} {reason} What I can do is outline {alt}.",
        "I'm sorry, but I cannot provide {task} for \"{request_excerpt}\". {reason} If useful, I can switch to {alt}.",
        "I must step back from that request. When a prompt is framed as \"{request_excerpt}\", {note} {reason} I can still provide {alt}.",
    ]
    return choose(rng, templates).format(
        request_excerpt=request_excerpt,
        task=task,
        reason=reason,
        alt=alt,
        note=note,
    )


def rewrite_dataset(dataset_name, path):
    rewriter = {
        "jailbreak_removal": rewrite_jailbreak_output,
        "over_refusal_injection": rewrite_over_refusal_output,
    }[dataset_name]

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    total = 0
    with open(path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
        for index, line in enumerate(src, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            example["output"] = rewriter(example, index)
            dst.write(json.dumps(example, ensure_ascii=False))
            dst.write("\n")
            total += 1

    os.replace(tmp_path, path)
    return total


def main():
    parser = argparse.ArgumentParser(description="Deterministically diversify refusal-style training outputs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DEFAULT_DATASETS),
        default=sorted(DEFAULT_DATASETS),
        help="Datasets to rewrite in place.",
    )
    args = parser.parse_args()

    for dataset_name in args.datasets:
        path = DEFAULT_DATASETS[dataset_name]
        total = rewrite_dataset(dataset_name, path)
        print(f"rewrote {dataset_name}: {total} rows -> {path}")


if __name__ == "__main__":
    main()
