# 크립토 추천 스크리너 자동화 (GitHub Actions + Telegram)

컴퓨터를 꺼둬도 매일 정해진 시간에 자동으로 업비트 코인을 분석해서,
대형/중형/소형 그룹별 추천 코인을 계산하고 **텔레그램으로 나에게만** 메시지를 보내주는
자동화입니다. 서버나 SQL 데이터베이스를 직접 운영할 필요는 없습니다 — GitHub이 대신
실행해주고, 결과는 `predictions.json` 파일 하나로 저장/추적합니다.

## 준비물 (전부 무료)

1. GitHub 계정 (없으면 github.com 에서 무료로 가입)
2. 핸드폰에 **Telegram** 앱 설치

## 설정 순서

### 1) 텔레그램 봇 만들기
1. 텔레그램에서 **@BotFather** 를 검색해서 대화를 시작합니다. (파란 체크 표시가 있는 공식 봇)
2. `/newbot` 명령을 보냅니다.
3. 봇 이름(아무거나, 예: `내 크립토 알리미`)과 봇 아이디(영문, `xxx_bot`으로 끝나야 함,
   예: `my_crypto_reco_bot`)를 순서대로 입력합니다.
4. 완료되면 BotFather가 **토큰(token)**을 줍니다.
   `123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` 같은 형태입니다.
   이게 `TELEGRAM_BOT_TOKEN` 값입니다. 잘 복사해두세요.

### 2) 내 chat_id 알아내기
1. 방금 만든 내 봇을 텔레그램에서 검색해서(아이디로 검색) 대화를 열고 **/start** 또는
   아무 메시지나 하나 보냅니다. (봇이 먼저 나에게 말을 걸 수는 없어서, 내가 먼저 말을 걸어야 합니다)
2. 브라우저에서 아래 주소를 열어봅니다 (TOKEN 부분을 1단계에서 받은 토큰으로 교체):
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
3. 결과 안에서 `"chat":{"id":123456789, ...}` 같은 부분을 찾습니다.
   그 숫자가 `TELEGRAM_CHAT_ID` 값입니다.
   (아무것도 안 보이면 봇에게 메시지를 보낸 다음 새로고침 해보세요.)

### 3) 이 폴더를 GitHub 저장소로 만들기
1. github.com에서 새 저장소(Repository)를 만듭니다. Public도 되고 Private도 됩니다.
2. 이 폴더(`crypto_screener.py`, `requirements.txt`, `predictions.json`, `README.md`,
   `.github/workflows/daily-screener.yml`)를 통째로 그 저장소에 업로드합니다.

### 4) GitHub 저장소에 토큰/채팅ID 등록하기
1. 저장소 페이지 → **Settings** → **Secrets and variables** → **Actions**
2. **New repository secret** 을 두 번 눌러서 아래 두 개를 각각 등록합니다:
   - Name: `TELEGRAM_BOT_TOKEN`, Value: 1단계에서 받은 토큰
   - Name: `TELEGRAM_CHAT_ID`, Value: 2단계에서 찾은 숫자

### 5) 실행 시간 확인/조정 (선택)
`.github/workflows/daily-screener.yml` 파일의 이 줄:
```
- cron: '0 22 * * *'
```
는 UTC 기준 22:00 = **한국시간 아침 7시**에 실행되도록 되어 있습니다.
다른 시간을 원하시면 앞의 숫자 두 개(분 시)를 바꾸시면 됩니다.
(예: 한국시간 밤 9시에 받고 싶다면 UTC 12:00 → `'0 12 * * *'`)

### 6) 첫 실행 테스트
1. 저장소의 **Actions** 탭 → 왼쪽에서 "Daily Crypto Screener" 선택
2. **Run workflow** 버튼으로 지금 바로 한 번 수동 실행해서 텔레그램 메시지가 잘 오는지 확인
3. 잘 오면 이후로는 매일 자동으로 실행됩니다

## 이 자동화가 하는 일

1. 업비트 원화 마켓 전체 코인을 가져와서, 시가총액(CoinGecko 기준)으로
   대형/중형/소형으로 나눕니다.
2. 거래대금 하위 종목(거래 저조)은 스캔에서 제외합니다.
3. 그룹별로 최대 `PER_GROUP_CAP`개(기본 15개)씩 몬테카를로 시뮬레이션 +
   추세 분석을 돌리고, 거래량 확인·BTC 대비 상대강도·기간별 추세 정합
   세 가지 보정 요인을 반영해 점수를 매깁니다.
4. 그룹별 TOP5를 뽑고, 조건을 만족하는 코인엔 "✓추천" 표시를 붙입니다.
5. `predictions.json`에 그날의 추천을 기록하고, 7일 지난 과거 추천은
   실제 가격과 비교해 적중/불일치를 기록합니다.
6. 요약 메시지를 텔레그램 봇으로 나에게만 전송합니다.

## 한계 / 참고

- 이건 과거 가격 데이터 기반 통계 모델입니다. 투자 조언이 아니며,
  실제 상승을 보장하지 않습니다.
- `TELEGRAM_BOT_TOKEN`은 비밀번호와 같습니다. 절대 남에게 공유하거나
  공개 저장소의 코드 안에 직접 적지 마세요 (반드시 GitHub Secrets로만 등록).
- CoinGecko/업비트 공개 API에는 호출 빈도 제한이 있습니다. 하루 한 번 실행하는
  용도로는 문제없지만, `PER_GROUP_CAP`를 너무 크게 올리면 실행 시간이 길어지거나
  일시적으로 API가 막힐 수 있습니다.

