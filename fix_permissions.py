#!/usr/bin/env python3
"""
Fix permissions for Multi-Agent ML Integration System
Run this script if you encounter permission errors during startup
"""

import os
import tempfile
import stat

def fix_permissions():
    """Fix common permission issues"""
    print("🔧 Fixing permissions for Multi-Agent ML Integration System...")
    
    # 1. Check /tmp permissions
    tmp_dir = tempfile.gettempdir()
    print(f"📁 Temp directory: {tmp_dir}")
    
    try:
        # Test if we can create a directory in /tmp
        test_dir = os.path.join(tmp_dir, "mal_integration_test")
        os.makedirs(test_dir, exist_ok=True)
        os.rmdir(test_dir)
        print("✅ /tmp directory is writable")
    except PermissionError:
        print("❌ /tmp directory has permission issues")
        print("🔧 Setting up alternative directory...")
        
        # Create alternative directory in user's home
        alt_dir = os.path.expanduser("~/mal_integration_workspace")
        os.makedirs(alt_dir, exist_ok=True)
        
        # Set proper permissions
        os.chmod(alt_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
        
        print(f"✅ Created alternative directory: {alt_dir}")
        print(f"💡 Set environment variable: export MAL_STATES_DIR={alt_dir}")
        
        # Create .env file
        env_content = f"""# Multi-Agent ML Integration Environment
MAL_STATES_DIR={alt_dir}
# Add your Slack tokens here:
# SLACK_BOT_TOKEN=xoxb-your-token
# SLACK_APP_TOKEN=xapp-your-token
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ Created .env file with MAL_STATES_DIR")
    
    # 2. Check user_data directory
    user_data_dir = "user_data"
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir, exist_ok=True)
        print(f"✅ Created {user_data_dir} directory")
    
    # Set proper permissions for user_data
    try:
        os.chmod(user_data_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
        print(f"✅ Set permissions for {user_data_dir}")
    except:
        print(f"⚠️ Could not set permissions for {user_data_dir}")
    
    # 3. Check artifacts directory
    artifacts_dir = "artifacts"
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir, exist_ok=True)
        print(f"✅ Created {artifacts_dir} directory")
    
    print("\n🎉 Permission fixes complete!")
    print("\n📋 Next steps:")
    print("1. Add your Slack tokens to .env file")
    print("2. Run: python start_pipeline.py --mode slack")
    print("3. If issues persist, run with: export MAL_STATES_DIR=~/mal_integration_workspace")

if __name__ == "__main__":
    fix_permissions()
