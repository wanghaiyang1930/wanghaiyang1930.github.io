




**Image Content:**
```JSON
[
    {'type': 'text', 'text': 'Front left camera: '},
    {'type': 'text', 'text': 'frame 0 '},
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 1 '}
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 2 '}
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 3 '},
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'Front camera: '}
    {'type': 'text', 'text': 'frame 0 '},
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 1 '}
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 2 '}
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 3 '},
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'Front right camera: '}
    {'type': 'text', 'text': 'frame 0 '},
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 1 '}
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 2 '}
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 3 '},
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'Front telephoto camera: '},
    {'type': 'text', 'text': 'frame 0 '},
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 1 '}
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 2 '}
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
    {'type': 'text', 'text': 'frame 3 '},
    {'type': 'image', 'image': 'tenso(3, 1080, 1920)'},
]
```

## Prediction Message


**User Text**
```JSON
{
    "type": "text",
    "text": "<|traj_history_start|><|traj_history|>*48<|traj_history_end|><|route_start|>这里是导航信息<|route_end|>output the chain-of-thought reasoning of the driving process, then output the future trajectory."
}
```
**Message:**
```JSON
[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": user_text}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]
```

## Question & Anwser

**Message:**

```JSON
[
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a driving assistant that generates safe and accurate actions.",
            }
        ],
    },
    {
        "role": "user",
        "content": image_content + [{"type": "text", "text": "<|question_start|>这里是问题描述？<|question_end|>"}],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<|answer_start|>"}],
    },
]
```