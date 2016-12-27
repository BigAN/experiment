import requests

url = "http://apimobile.vip.sankuai.com/locate/v2/sdk/loc?client_source= waimairc_locate_ana"
body = '{"wifi_towers":[{"mac_address":"78:a1:06:91:29:f8","signal_strength":"-69","ssid":"888888"},{"mac_address":"9c:21:6a:2a:8c:d4","signal_strength":"-70","ssid":"hehui"},{"mac_address":"e4:d3:32:da:61:54","signal_strength":"-70","ssid":"TP-LINK700"},{"mac_address":"fc:d7:33:82:92:ca","signal_strength":"-73","ssid":"WEARE"},{"mac_address":"44:97:5a:2a:f9:3c","signal_strength":"-73","ssid":"FAST_F93C"},{"mac_address":"24:69:68:ca:dc:d8","signal_strength":"-75","ssid":"TP-LINK_WENT"},{"mac_address":"28:2c:b2:53:ec:66","signal_strength":"-75","ssid":""},{"mac_address":"78:a1:06:bb:60:6a","signal_strength":"-76","ssid":"TP-LINK_BB606A"},{"mac_address":"3c:1e:04:10:1d:2c","signal_strength":"-76","ssid":"D-Link_DIR-629"},{"mac_address":"f0:b4:29:2a:74:a3","signal_strength":"-76","ssid":"Xiaomi_101"},{"mac_address":"ec:26:ca:58:33:34","signal_strength":"-77","ssid":"TP-LINK_605"},{"mac_address":"0c:72:2c:be:ea:ec","signal_strength":"-78","ssid":"FAST_BEEAEC"},{"mac_address":"1c:fa:68:7d:64:48","signal_strength":"-79","ssid":"FAST_7D6448"},{"mac_address":"88:25:93:b6:27:7a","signal_strength":"-79","ssid":"dlink20130108"},{"mac_address":"6c:59:40:89:c1:ae","signal_strength":"-79","ssid":"MERCURY_C1AE"},{"mac_address":"38:83:45:cd:12:72","signal_strength":"-81","ssid":"JL"},{"mac_address":"a8:57:4e:57:97:48","signal_strength":"-81","ssid":"TP-LINK_579748"},{"mac_address":"54:89:98:5b:57:6b","signal_strength":"-81","ssid":"ChinaNet-rDIE"},{"mac_address":"bc:f6:85:9b:5a:10","signal_strength":"-81","ssid":"D-Link_DIR-600A"},{"mac_address":"88:25:93:57:f2:b8","signal_strength":"-83","ssid":"TP-LINK_20151005"},{"mac_address":"c8:3a:35:02:f8:f0","signal_strength":"-83","ssid":"Tenda_02F8F0"},{"mac_address":"1c:fa:68:0d:73:4a","signal_strength":"-84","ssid":"TP-LINK_301"},{"mac_address":"52:1e:93:7d:e0:c6","signal_strength":"-84","ssid":"ITV-AFC5"},{"mac_address":"cc:34:29:19:be:c4","signal_strength":"-84","ssid":"FAST_19BEC4"},{"mac_address":"8c:a6:df:2c:33:48","signal_strength":"-84","ssid":"TP-LINK_3348"},{"mac_address":"d4:83:04:94:ef:b8","signal_strength":"-85","ssid":"FAST_305"},{"mac_address":"00:25:86:43:59:a4","signal_strength":"-85","ssid":"yylx"},{"mac_address":"42:89:98:5b:57:6b","signal_strength":"-85","ssid":"aWiFi"},{"mac_address":"6c:e8:73:33:fc:a6","signal_strength":"-86","ssid":"TP-LINK_33FCA6"},{"mac_address":"bc:d1:77:ff:c6:c8","signal_strength":"-86","ssid":"TP-LINK_2.4GHz_FFC6C8"},{"mac_address":"8c:f2:28:a0:2a:78","signal_strength":"-86","ssid":"MERCURY_2A78"},{"mac_address":"8e:25:93:79:3d:33","signal_strength":"-86","ssid":"Guest_3D33"},{"mac_address":"fc:d7:33:52:06:ee","signal_strength":"-86","ssid":"TP-LINK_06EE"},{"mac_address":"74:1e:93:7d:e0:c5","signal_strength":"-86","ssid":"ChinaNet-AFC5"},{"mac_address":"bc:d1:77:91:22:fe","signal_strength":"-86","ssid":"FAST_202"},{"mac_address":"f0:b4:29:f5:8c:87","signal_strength":"-86","ssid":"Xiaomi_8C86"},{"mac_address":"c0:61:18:dc:d1:a6","signal_strength":"-86","ssid":"FAST_DCD1A6"},{"mac_address":"8c:f2:28:6d:ef:4a","signal_strength":"-86","ssid":"20161022"},{"mac_address":"dc:9c:9f:67:e3:63","signal_strength":"-87","ssid":"ChinaNet-vbCT"},{"mac_address":"f0:b4:29:d5:6f:0c","signal_strength":"-88","ssid":"520ZSY"},{"mac_address":"14:e6:e4:3e:6e:8a","signal_strength":"-88","ssid":"FAST_xu"},{"mac_address":"c0:61:18:54:7d:20","signal_strength":"-88","ssid":"MERCURY_501"},{"mac_address":"44:97:5a:05:ff:d6","signal_strength":"-88","ssid":"FAST_303"},{"mac_address":"f4:ee:14:35:a7:0e","signal_strength":"-88","ssid":"ooxx"},{"mac_address":"88:25:93:79:3d:33","signal_strength":"-89","ssid":"TP-LINK_3D33"}],"cell_toewers":[{"mobile_network_code":"0","cell_id":"84818948","mobile_country_code":"460","location_area_code":"20944"}],"appname":"com.sankuai.meituan","wifiage":"2","gps":{"glat":"31.954277","glng":"118.727027"},"request_address":true}'
import json

j_body = json.loads(body)
print type(j_body)

r = requests.post(url, data = body)
print r.text
