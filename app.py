import streamlit as st
from main import graph, HumanMessage

# Page configuration
st.set_page_config(
    page_title="X Content Generator",
    page_icon="🐦",
    layout="centered"
)

# Main title
st.title("X Content Generator")

# Main content area
topic = st.text_area(
    "What would you like to post about?",
    placeholder="Example: Create 3 posts about Model Context Protocol (MCP)...",
    height=100
)

if st.button("Generate Posts"):
    if topic:
        # Create initial state
        initial_state = {
            "messages": [
                HumanMessage(content=topic)
            ]
        }
        
        # Run the graph
        with st.spinner("Generating..."):
            # Store the last generated content
            latest_posts = None
            
            for output in graph.stream(initial_state):
                message_type = "generate" if "generate" in output else "reflect"
                content = output[message_type]["messages"][-1].content
                
                if message_type == "generate":
                    latest_posts = content
            
            # Show only the last generated content
            if latest_posts:
                st.markdown("### Generated Posts")
                
                # Split content into individual posts
                posts = latest_posts.split("[POST")
                for i, post in enumerate(posts[1:], 1):
                    # Extract post content
                    post_content = post.split("]")[1].strip()
                    
                    # Display post
                    st.markdown(f"**Post {i}:**")
                    st.markdown(post_content)
    else:
        st.error("Please enter a topic first!")

# Simple usage instructions
with st.expander("How to use"):
    st.markdown("""
    1. Enter your topic in the text area above
    2. Click 'Generate Posts'
    3. Review the generated posts
    """) 