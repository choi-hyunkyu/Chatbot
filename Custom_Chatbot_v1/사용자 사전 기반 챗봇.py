# !pip install PyKomoran
from PyKomoran import *
from collections import OrderedDict, Counter
import pandas as pd
import json

'''
komoran model setting
'''
komoran = Komoran(model_path = DEFAULT_MODEL['FULL'])
komoran.set_user_dic('./data/dic/user.dic')

'''
menu list loading
'''
PATH = './data/dic/menu_list/'
burger = pd.read_csv(PATH + 'burger.txt')['burger'].tolist()
beverage = pd.read_csv(PATH + 'beverage.txt')['beverage'].tolist()
size = pd.read_csv(PATH + 'size.txt')['size'].tolist()
menu_type = pd.read_csv(PATH + 'menu_type.txt')['menu_type'].tolist()
dessert = pd.read_csv(PATH + 'dessert.txt')['dessert'].tolist()
icecreem = pd.read_csv(PATH + 'icecreem.txt')['icecreem'].tolist()
morning = pd.read_csv(PATH + 'morning.txt')['morning'].tolist()

'''
save as json file, intent list for customer in store
'''
menu_list = OrderedDict()
menu_list['burger'] = burger
menu_list['beverage'] = beverage
menu_list['size'] = size
menu_list['menu_type'] = menu_type
menu_list['dessert'] = dessert
menu_list['icecreem'] = icecreem
menu_list['morning'] = morning

with open(PATH + 'menu_list.json', 'w', encoding = 'utf-8') as file:
    json.dump(menu_list, file, ensure_ascii = False, indent = '\t')

'''
data loading
'''
with open('./data/dic/menu_list/menu_list.json', 'r', encoding = 'utf-8') as file:
    intent_dict = json.load(file)

'''
functions
'''
# preprocessing number to string
def num2str(order_dict):
    for index, value in enumerate(order_dict):
        if order_dict[index] == '0':
            order_dict[index] = '영'
        elif order_dict[index] == '1':
            order_dict[index] = '일'
        elif order_dict[index] == '2':
            order_dict[index] = '이'
        elif order_dict[index] == '3':
            order_dict[index] = '삼'
        elif order_dict[index] == '4':
            order_dict[index] = '사'
        elif order_dict[index] == '5':
            order_dict[index] = '오'
        elif order_dict[index] == '6':
            order_dict[index] = '육'
        elif order_dict[index] == '7':
            order_dict[index] = '칠'
        elif order_dict[index] == '8':
            order_dict[index] = '팔'
        elif order_dict[index] == '9':
            order_dict[index] = '구'
    
    return order_dict

# extract intent from customer
def extract_intent(input_sentence):
    '''
    str2dict with customer
    '''
    order_dict = {index: value for index, value in enumerate([i for i in input_sentence])}
    tkn_order = "".join(list(num2str(order_dict).values()))
    customer_order = {'CUSTOMER_ORDER': komoran.nouns(tkn_order)}

    for index, samples in enumerate(customer_order.items()):
        key, value = samples
        for index, word in enumerate(value):
            if '안녕' in word:
                del value[index]

    customer_intent_count = Counter(value)

    return customer_order, customer_intent_count

# matching
def matching_menu(order, intent_dict):
    customer_dict = {}
    for index, items in enumerate(intent_dict.items()): # v: list
        category, menues = items
        menu = [value for value in menues if value in order]
        customer_dict.update({category: menu})

    return customer_dict

# response
def response(customer_dict):
    # 기본 햄버거 주문
    if len(customer_dict['burger']) != 0 and len(customer_dict['morning']) == 0:
        print("요청하신 {}, {}, {}-{} {} {} 주문이 완료되었습니다.".format(
            customer_dict['burger'],
            customer_dict['beverage'],
            customer_dict['size'],
            customer_dict['menu_type'],
            customer_dict['dessert'],
            customer_dict['icecreem']
            ))

    # 기본 모닝 주문
    elif len(customer_dict['burger']) == 0 and len(customer_dict['morning']) != 0:
        print("요청하신 {}, {}, {}-{} {} {} 주문이 완료되었습니다.".format(
            customer_dict['morning'],
            customer_dict['beverage'],
            customer_dict['size'],
            customer_dict['menu_type'],
            customer_dict['dessert'],
            customer_dict['icecreem']
            ))

    # 햄버거, 모닝 외 주문
    elif len(customer_dict['burger']) == 0 and len(customer_dict['morning']) == 0:
        print("요청하신 {} {} {} 주문이 완료되었습니다.".format(
            customer_dict['beverage'],
            customer_dict['dessert'],
            customer_dict['icecreem']
            ))
    # 햄버거, 모닝 동시 주문 시
    elif len(customer_dict['burger']) != 0 and len(customer_dict['morning']) != 0:
        print("손님, 요청하신 {} 제품과 {} 제품은 동시에 주문이 어렵습니다. 다른 제품을 주문해주세요.".format(
        customer_dict['burger'],
        customer_dict['morning']
        ))

# order menu
def chatting(input_sentence, intent_dict):
    customer_intents, counts = extract_intent(input_sentence)
    # dict2list with store
    store_list = [intent for categories, menues in intent_dict.items() for intent in menues]
    # CUSTOMER's order list
    order = [value for value in store_list if value in customer_intents['CUSTOMER_ORDER']]
    # Matching between CUSTOMER's order and menues in the store
    customer_dict = matching_menu(order, intent_dict)
    # Answer
    #print("요청하신 {}, {}, {}{} 주문이 완료되었습니다.".format(customer_dict['burger'], customer_dict['beverage'], customer_dict['size'], customer_dict['menu_type']))
    response(customer_dict)

    return customer_dict

# chat with bot
def chat(intent_dict):
    input_sentence = ""
    while(1):
        try:
            '''
            주문 접수 여부
            '''
            input_sentence = input('주문하시겠습니까? > ')
            if input_sentence == '아니오' or input_sentence == '아니':
                print("감사합니다. 안녕히 가십시오.")
                break

            '''
            주문 접수
            '''
            customer_dict = chatting(input_sentence, intent_dict)

            '''
            주문 종료
            '''
            check_sentence = input('주문을 이어서 하시겠습니까? > ')
            if check_sentence == '아니오' or check_sentence == '아니':
                print("주문이 완료되었습니다.")
                break
            elif check_sentence == '예' or check_sentence == '네': continue

        except:
            print("Error")

    return customer_dict

'''
Result 1
'''
input_sentence_1 = "안녕하세요, 상하이버거 미디움세트, 빅맥버거 라지세트에 음료수는 아이스커피, 콜라로 주세요."
input_sentence_2 = "상하이버거 라지세트하나에 빅맥단품으로 주세요. 아 그리고 상하이 버거에는 콜라로 주세요"
input_sentence_3 = "빅맥 단품 하나 주세요."
input_sentence_4 = "더블1955버거 단품 하나 주세요."
input_sentence_5 = "슈슈버거 세트에 음료수는 환타로 주세요."
input_sentence_6 = "소세지 에그 맥머핀 세트하나에 음료수는 배 칠러로 주세요. 아 그리고 아이스크림콘도 같이 주세요."
input_sentence_7 = "핫케익3개 주시면 감사하겠습니다." # 수량 출력 수정해야됨

customer_dict = chatting(input_sentence_4, intent_dict)
chatting(input_sentence_7, intent_dict)

'''
Result 2
'''
chat(intent_dict)
