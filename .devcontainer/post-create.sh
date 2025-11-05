#!/bin/bash

# Post-create script for dev container setup
echo "Running post-create setup..."

# Install Python packages
echo "Installing Python packages..."
pip3 install -r requirements.txt

# Install MNIST project requirements (if needed for other lessons)
if [ -f "MNIST/requirements.txt" ]; then
    echo "Installing MNIST project requirements..."
    pip3 install -r MNIST/requirements.txt
else
    echo "Note: MNIST/requirements.txt not found, skipping..."
fi

# Install Claude Code globally
echo "📦 Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

# Set up environment variables
echo "Setting up environment variables..."

# Path to secrets file
SECRETS_FILE=".devcontainer/secrets.env"

# Check if secrets file exists
if [ -f "$SECRETS_FILE" ]; then
    echo "Loading environment variables from secrets.env..."
    # Source the secrets file to load the environment variables
    source "$SECRETS_FILE"

    # Export Azure environment variables
    export AZURE_AI_PROJECT_ENDPOINT
    export AZURE_OPENAI_ENDPOINT
    export AZURE_TENANT_ID
    export AZURE_CLIENT_ID
    export AZURE_CLIENT_SECRET

    # Add them to both bash and zsh profiles so they persist across all shells
    for profile in ~/.bashrc ~/.zshrc; do
        echo "" >> "$profile"
        echo "# Azure AI Foundry Environment Variables" >> "$profile"

        if [ ! -z "$AZURE_AI_PROJECT_ENDPOINT" ]; then
            echo "export AZURE_AI_PROJECT_ENDPOINT=\"$AZURE_AI_PROJECT_ENDPOINT\"" >> "$profile"
        fi

        if [ ! -z "$AZURE_OPENAI_ENDPOINT" ]; then
            echo "export AZURE_OPENAI_ENDPOINT=\"$AZURE_OPENAI_ENDPOINT\"" >> "$profile"
        fi

        if [ ! -z "$AZURE_TENANT_ID" ]; then
            echo "export AZURE_TENANT_ID=\"$AZURE_TENANT_ID\"" >> "$profile"
        fi

        if [ ! -z "$AZURE_CLIENT_ID" ]; then
            echo "export AZURE_CLIENT_ID=\"$AZURE_CLIENT_ID\"" >> "$profile"
        fi

        if [ ! -z "$AZURE_CLIENT_SECRET" ]; then
            echo "export AZURE_CLIENT_SECRET=\"$AZURE_CLIENT_SECRET\"" >> "$profile"
        fi
    done

    echo "Environment variables set successfully!"
else
    echo "Warning: secrets.env file not found at $SECRETS_FILE"
    echo "Please copy secrets.env.template to secrets.env and fill in your Azure credentials."
fi

# Display Azure login instructions
echo ""
echo "=========================================="
echo "Azure Authentication Setup"
echo "=========================================="
echo "To authenticate with Azure AI Foundry, run:"
echo "  az login"
echo ""
echo "This will open a browser for you to sign in with your Azure account."
echo "The DefaultAzureCredential will automatically use these credentials."
echo "=========================================="
echo ""

echo "Post-create setup complete!"
