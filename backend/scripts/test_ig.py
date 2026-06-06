import instaloader
L = instaloader.Instaloader(
    download_videos=False,
    download_video_thumbnails=False,
    save_metadata=False,
    quiet=True,
)
for url in [
    'https://www.instagram.com/reel/C3tK9bHJW1n/',
    'https://www.instagram.com/p/CzKz9sJMi4N/',
    'https://www.instagram.com/reel/C5pQ8qDuY7W/',
]:
    try:
        shortcode = url.rstrip('/').split('/')[-1].split('?')[0]
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        print(f'PASS {url}')
        print(f'  owner:   {post.owner_username}')
        print(f'  likes:   {post.likes}  comments: {post.comments}  views: {post.video_view_count}')
        print(f'  caption: {(post.caption or "")[:120]}')
    except Exception as e:
        print(f'FAIL {url}: {str(e)[:200]}')
