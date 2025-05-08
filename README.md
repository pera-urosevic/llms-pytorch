# LLM examples

## 1 - DirectML
Basic test for torch_directml with early stop, predicts f(x) = x * 2 - 1. Outputs: training time; predicted value, correct value, and difference.

```
Epoch: 0 | Loss: 17.1527
Epoch: 1 | Loss: 14.2939
...
Epoch: 58 | Loss: 2.1062
Epoch: 59 | Loss: 2.1062

Time: 0.58s

Predicted: 19.111679077148438, Correct: 19, Difference: 0.1116790771484375
```

## 2 - DialoGPT
Example for basic LLM usage with DialoGPT, which is based on an old LLM, GPT-2. Answers are randomized via:
```
top_p=0.9
temperature=0.7
do_sample=True
```

Good answer:
> User: Hi!  
> Bot: Hello! How can I assist you today?
>
> User: Tell me the capital of France.  
> Bot: The capital of France is Paris.

Bad answer:
> User: Hi!  
> Bot: Hello! How can I assist you today?
> 
> User: Tell me the capital of France.  
> Bot: I'm sorry, I'm not available today.

## 3 - Llama
An interactive example for basic LLM usage with Llama 3.2, a much better model than DialoGPT.

> User:  What is a triangle?  
> Assistant: A triangle is a polygon with three sides and three angles. It is a fundamental shape in geometry and has numerous applications in various fields, such as architecture, engineering, and art. The angles of a triangle add up to 180 degrees.
> 
> User: What are the types of triangles?  
> Assistant: There are three main types of triangles:
> 1. Equilateral Triangle: All sides are equal in length.
> 2. Isosceles Triangle: Two sides are equal in length, and the other two angles are equal.
> 3. Scalene Triangle: All sides are unequal in length.
> 
> User: And types by angles?  
> Assistant: There are three types of triangles based on the measure of their angles:
> 1. Acute Triangle: All angles are less than 90 degrees.
> 2. Right Triangle: One angle is exactly 90 degrees.
> 3. Obtuse Triangle: One angle is greater than 90 degrees.

## 4 - Context
For small datasets, it is possible to add it as part of a chat context. Size depends on model, e.g. for Llama 3.2 it's 128K.

In this example, dataset is a paragraph from book The Adventures of Sherlock Holmes, where name Count Von Kramm was changed to Count Von Krammzzz to exclude any references already present in the model.

**[Without Context] Answer:**  
I couldn't find any information on a person named Count Von Krammzzz. It's possible that you may be thinking of a different name or that this is a fictional character. Could you provide more context or clarify who Count Von Krammzzz is? I'll do my best to help.

Supplied context:
> An excerpt from the book Sherlock Holmes: “You may address me as the Count Von Krammzzz, a Bohemian nobleman. I understand that this gentleman, your friend, is a man of honour and discretion, whom I may trust with a matter of the most extreme importance. If not, I should much prefer to communicate with you alone.”

**[With Context] Answer:**  
Count Von Krammzzz is a character from the Sherlock Holmes stories created by Sir Arthur Conan Doyle. He is a wealthy and influential nobleman who often provides Sherlock Holmes with information and assistance in his cases. He is known for his aristocratic manner and his eccentricities, and is often portrayed as a somewhat eccentric and flamboyant character.

## 5 - RAG
Like previous example, character Count Von Kramm has been renamed to Count Von Krammzzz to differentiate from base model references.

**Data preparation:**
```
Collection 'the_adventures_of_sherlock_holmes' is empty. Proceeding with data preparation.
Loaded data successfully, size: 581571 characters.
Split text into 1445 chunks.
  Processed batch 10/46
  Processed batch 20/46
  Processed batch 30/46
  Processed batch 40/46
Embeddings generated successfully for 1445 chunks.
Adding 1445 records to ChromaDB collection 'the_adventures_of_sherlock_holmes'...
  Added batch 1/1
Successfully added records. Collection count: 1445
Data preparation finished in 3.05 seconds
```

**Test for base model knowledge, should be ignored**

> User: who is count von kramm  
> Assistant: There is no mention of a person named Count Von Kramm in the passage provided.

**Test for RAG knowledge**

> User: who is count von krammzzz  
> Assistant: The Count von Krammzzz is a fictional character in the Sherlock Holmes stories by Sir Arthur Conan Doyle. He is a Bohemian nobleman, a member of a noble family, and a friend of Sherlock Holmes.

**Test for random book details**
> User: where did sherlock lived?  
> Assistant: Sherlock Holmes lived at 221B Baker Street in London, England.

## 6 - LangChain
Using LangChain framework to simplify and streamline RAG process.

**Example 1. Asking for knowledge outside of topic**
> User:
> who is count von count
> 
> Context:  
> “Pray take a seat,” said Holmes. “This is my friend and colleague, Dr. Watson, who is occasionally good enough to help me in my cases. Whom have I the honour to address?”  “You may address me as the Count von Krammzzz, a Bohemian nobleman. I understand that this gentleman, your friend, is a man of honour and discretion, whom I may trust with a matter of the most extreme importance. If not, I should much prefer to communicate with you alone.”  I rose to go, but Holmes caught me by the wrist and p...
> 
> Assistant:  
> I don't know.

**Example 2. Asking about a character name that exists only in the provided context, but not in the real book**
> User: 
> how does count von krammzzz present himself
> 
> Context:
> “Pray take a seat,” said Holmes. “This is my friend and colleague, Dr. Watson, who is occasionally good enough to help me in my cases. Whom have I the honour to address?”  “You may address me as the Count von Krammzzz, a Bohemian nobleman. I understand that this gentleman, your friend, is a man of honour and discretion, whom I may trust with a matter of the most extreme importance. If not, I should much prefer to communicate with you alone.”  I rose to go, but Holmes caught me by the wrist and p...
> 
> Assistant: 
> Count von Krammzzz presents himself as a Bohemian nobleman, using a formal title and a somewhat flamboyant and aristocratic tone.

**Example 3. A common type of question**
> User:  
> where was irene adler born
> 
> Context:  
> “But you can understand,” said our strange visitor, sitting down once more and passing his hand over his high white forehead, “you can understand that I am not accustomed to doing such business in my own person. Yet the matter was so delicate that I could not confide it to an agent without putting myself in his power. I have come _incognito_ from Prague for the purpose of consulting you.”  “Then, pray consult,” said Holmes, shutting his eyes once more.  “The facts are briefly these: Some five ye...
> 
> Assistant:  
> Irene Adler was born in New Jersey in the year 1858.
