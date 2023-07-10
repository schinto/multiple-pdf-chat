css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1.5rem; display: flex
}
.chat-message.user {
    background-color: #f0ead6; /* light yellow */
}
.chat-message.bot {
    background-color: #e6e9f0; /* light blue */
}
.chat-message .avatar {
    width: 20%;
    flex-shrink: 0; /* Prevents shrinking of the avatar, keeping it at its set width */
}
.chat-message .avatar img {
    max-width: 50px;
    max-height: 50px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #333333; /* dark gray */
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="./app/static/snoopy.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="./app/static/woodstock.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""
