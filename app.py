
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import time
import pathlib

# --- Connect4環境 ---
class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype="int32")
        self.player = 1

    def get_player(self):
        return self.player

    def reset(self):
        self.board = np.zeros((6, 7), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        for i in range(6):
            if self.board[i][action] == 0:
                self.board[i][action] = self.player
                break
        done = self.have_winner() or len(self.legal_actions()) == 0
        reward = 1 if self.have_winner() else 0
        self.player *= -1
        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = np.where(self.board == 1, 1, 0)
        board_player2 = np.where(self.board == -1, 1, 0)
        return np.array([board_player1, board_player2], dtype="int32").flatten()

    def legal_actions(self):
        return [i for i in range(7) if self.board[5][i] == 0]

    def have_winner(self):
        p = self.player
        for i in range(4):
            for j in range(6):
                if (
                    self.board[j][i] == p and
                    self.board[j][i + 1] == p and
                    self.board[j][i + 2] == p and
                    self.board[j][i + 3] == p
                ):
                    return True
        for i in range(7):
            for j in range(3):
                if (
                    self.board[j][i] == p and
                    self.board[j + 1][i] == p and
                    self.board[j + 2][i] == p and
                    self.board[j + 3][i] == p
                ):
                    return True
        for i in range(4):
            for j in range(3):
                if (
                    self.board[j][i] == p and
                    self.board[j + 1][i + 1] == p and
                    self.board[j + 2][i + 2] == p and
                    self.board[j + 3][i + 3] == p
                ):
                    return True
        for i in range(4):
            for j in range(3, 6):
                if (
                    self.board[j][i] == p and
                    self.board[j - 1][i + 1] == p and
                    self.board[j - 2][i + 2] == p and
                    self.board[j - 3][i + 3] == p
                ):
                    return True
        return False

# --- ニューラルネットワーク ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(84, 336)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(336, 336)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(336, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# --- モデルロード ---
@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data['model'])
    model.eval()
    return model, device

# --- 2つのモデルをロード ---
def load_models():
    model_first, device = load_model("model_first.cpt")
    model_second, _ = load_model("model_second.cpt")
    return model_first, model_second, device

# --- 状態価値を返す ---
def get_state_value(state, player, model_first, model_second, device):
    model = model_first if player == 1 else model_second
    X = np.array([state], dtype=np.float32)
    with torch.no_grad():
        tX = torch.from_numpy(X).to(device)
        value = model(tX)
        return float(value.item())

# --- AIによる各Actionの推奨度を取得 ---
def predict(env, model_first, model_second, device):
    actions = env.legal_actions()
    values = []
    current_player = env.player
    for a in range(7):
        if a in actions:
            tmp_board = env.board.copy()
            tmp_player = env.player
            for i in range(6):
                if tmp_board[i][a] == 0:
                    tmp_board[i][a] = tmp_player
                    break
            board_player1 = np.where(tmp_board == 1, 1, 0)
            board_player2 = np.where(tmp_board == -1, 1, 0)
            flat = np.array([board_player1, board_player2], dtype="int32").flatten()
            v = get_state_value(flat, current_player, model_first, model_second, device)
            values.append(v)
        else:
            values.append(-9999)
    return np.array(values)

# --- 盤面表示 ---
def display_board(board):
    header = "１２３４５６７"
    board_lines = []
    for row in reversed(range(6)):
        row_str = ""
        for col in range(7):
            if board[row, col] == 1:
                row_str += "●"
            elif board[row, col] == -1:
                row_str += "✕"
            else:
                row_str += "□"
        board_lines.append(row_str)
    board_body = "<br/>".join(board_lines)
    html = f'<span style="font-family:\'MS Gothic\',\'Osaka-Mono\',monospace;font-size:24px;line-height:1.1;">{header}<br/>{board_body}</span>'
    st.markdown(html, unsafe_allow_html=True)

# --- Streamlitアプリ本体 ---
st.title("Connect4（pytorch版V2）")
try:
    model_first, model_second, device = load_models()
except Exception as e:
    st.error(f"モデルのロードに失敗しました: {e}")
    st.stop()

# セッション初期化
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'first_player' not in st.session_state:
    st.session_state.first_player = None

def start_new_game(first_player):
    st.session_state.game = Connect4()
    if first_player == 'ai':
        st.session_state.game.player = -1
    else:
        st.session_state.game.player = 1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.message = "あなたの番です。下のボタンから列を選んでください。" if first_player == 'human' else "AIが先手です。"
    st.session_state.ai_scores = None
    st.session_state.game_started = True
    st.session_state.first_player = first_player

# ゲーム開始前：先手/後手選択
if not st.session_state.game_started:
    st.info("先手（人間）か後手（AI）を選んでください。")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("先手（人間）で始める", key="start_human"):
            start_new_game('human')
            st.rerun()
    with col2:
        if st.button("後手（AI）で始める", key="start_ai"):
            start_new_game('ai')
            st.rerun()
    st.stop()

# ゲーム進行中
display_board(st.session_state.game.board)
message_placeholder = st.empty()
message_placeholder.info(st.session_state.message)

if st.session_state.game_over:
    if st.session_state.winner == 1:
        st.success("🎉 あなたの勝ちです！おめでとうございます！")
    elif st.session_state.winner == -1:
        st.error("🤖 AIの勝ちです。")
    elif st.session_state.winner == 0:
        st.warning("引き分けです。")
    else:
        st.info("ゲーム終了")

    if st.button("新しいゲームを始める"):
        st.session_state.game_started = False
        st.session_state.first_player = None
        st.rerun()
else:
    current_player = st.session_state.game.get_player()
    if current_player == 1:  # 人間のターン
        legal_actions = st.session_state.game.legal_actions()
        cols = st.columns(7)
        for i in range(7):
            with cols[i]:
                if st.button(f"{i+1}", key=f"col_{i}", disabled=(i not in legal_actions)):
                    _, reward, done = st.session_state.game.step(i)
                    if done:
                        st.session_state.game_over = True
                        if reward == 1:
                            st.session_state.winner = 1  # Human
                        elif len(st.session_state.game.legal_actions()) == 0:
                            st.session_state.winner = 0  # Draw
                        else:
                            st.session_state.winner = None
                    st.session_state.ai_scores = None
                    st.rerun()
    else:  # AIのターン
        st.session_state.message = "AIが思考中です..."
        message_placeholder.info(st.session_state.message)
        with st.spinner("AIが思考中..."):
            time.sleep(1.0)
            predicts = predict(st.session_state.game, model_first, model_second, device)
            legal_actions = st.session_state.game.legal_actions()
            # 合法手の中で最大値
            best_action = int(np.argmax([predicts[a] if a in legal_actions else -9999 for a in range(7)]))
            st.session_state.ai_scores = {a: predicts[a] for a in legal_actions}
            if best_action in legal_actions:
                _, reward, done = st.session_state.game.step(best_action)
                if done:
                    st.session_state.game_over = True
                    if reward == 1:
                        st.session_state.winner = -1  # AI
                    elif len(st.session_state.game.legal_actions()) == 0:
                        st.session_state.winner = 0  # Draw
                    else:
                        st.session_state.winner = None
            st.session_state.message = "あなたの番です。下のボタンから列を選んでください。"
            st.rerun()
