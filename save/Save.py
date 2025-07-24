import csv
from typing import Dict
import lxml.etree as ET
import datetime


def save_data_to_csv(file_path, headers, data):
    with open(file_path, 'w+') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        for row_data in data:
            f_csv.writerow(row_data.tolist())


def create_datetime_node(root_node: ET.Element):
    save_time = datetime.datetime.now()
    gen_dt_date = ET.SubElement(root_node, 'Create_Datetime', {'type': 'Date'})
    gen_dt_year = ET.SubElement(gen_dt_date, 'Year')
    gen_dt_year.text = str(save_time.year)
    gen_dt_month = ET.SubElement(gen_dt_date, 'Month')
    gen_dt_month.text = str(save_time.month)
    gen_dt_day = ET.SubElement(gen_dt_date, 'Day')
    gen_dt_day.text = str(save_time.day)
    gen_dt_time = ET.SubElement(root_node, 'Create_Datetime', {'type': 'Time'})
    gen_dt_hour = ET.SubElement(gen_dt_time, 'Hour')
    gen_dt_hour.text = str(save_time.hour)
    gen_dt_minute = ET.SubElement(gen_dt_time, 'Minute')
    gen_dt_minute.text = str(save_time.minute)
    gen_dt_second = ET.SubElement(gen_dt_time, 'Second')
    gen_dt_second.text = str(save_time.second)


def create_layer_node(root_node: ET.Element, layer: Dict):
    layer_units = ET.SubElement(root_node, 'units')
    layer_units.text = str(layer['units'])
    layer_func = ET.SubElement(root_node, 'activate')
    layer_func.text = layer['activate']


def save_net_structure_to_xml(file_path: str, mission_name: str, input_block: Dict, middle_block: Dict,
                              output_block: Dict):
    """
    保存网络结构到xml文件
    :param mission_name:
    :param file_path:
    :param input_block:
    :param middle_block:
    :param output_block:
    :return: 网络结构文件保存路径
    """
    with open(file_path, 'w+', newline='') as f:
        root = ET.Element('data')
        # 写入任务名称
        net_mission_name = ET.SubElement(root, 'Mission_name')
        net_mission_name.text = mission_name
        # 写入创建时间
        create_datetime_node(root)
        # 写入结构
        net_structure = ET.SubElement(root, 'Network_Structure')
        # 输入层
        input_layer_structure = ET.SubElement(net_structure, 'Input_Block')
        input_layer_structure_dim = ET.SubElement(input_layer_structure, 'dim')
        input_layer_structure_dim.text = str(input_block['dim'])
        create_layer_node(input_layer_structure, input_block)
        # 中间层
        middle_layers_structure = ET.SubElement(net_structure, 'Middle_Block')
        middle_layers_structure_num = ET.SubElement(middle_layers_structure, 'num')
        middle_layers_structure_num.text = str(middle_block['num'])
        for i in range(middle_block['num']):
            mdl_layer_str = 'Layer_%d' % i
            mdl_layer_i = ET.SubElement(middle_layers_structure, mdl_layer_str)
            create_layer_node(mdl_layer_i, {'units': middle_block['units'][i], 'activate': middle_block['activate'][i]})
        # 输出层
        output_layer_structure = ET.SubElement(net_structure, 'Output_Block')
        create_layer_node(output_layer_structure, {'units': output_block['dim'], 'activate': output_block['activate']})
        # 保存数据
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True, pretty_print=True)
    return file_path


def save_critic_structure_to_xml(file_path: str, mission_name: str, input_block: Dict, middle_block: Dict,
                                 output_block: Dict):
    """
    保存网络结构到xml文件
    :param mission_name:
    :param file_path:
    :param input_block:
    :param middle_block:
    :param output_block:
    :return: 网络结构文件保存路径
    """
    with open(file_path, 'w+', newline='') as f:
        root = ET.Element('data')
        # 写入任务名称
        net_mission_name = ET.SubElement(root, 'Mission_name')
        net_mission_name.text = mission_name
        # 写入创建时间
        create_datetime_node(root)
        # 写入结构
        net_structure = ET.SubElement(root, 'Network_Structure')
        # 输入层
        input_layer_structure = ET.SubElement(net_structure, 'Input_Block')
        input_state_block = ET.SubElement(input_layer_structure, 'Input_State_Block')
        input_state_block_dim = ET.SubElement(input_state_block, 'dim')
        input_state_block_dim.text = str(input_block['state']['dim'])
        create_layer_node(input_state_block, input_block['state'])
        input_action_block = ET.SubElement(input_layer_structure, 'Input_Action_Block')
        input_action_block_dim = ET.SubElement(input_action_block, 'dim')
        input_action_block_dim.text = str(input_block['action']['dim'])
        create_layer_node(input_action_block, input_block['action'])
        # 中间层
        middle_layers_structure = ET.SubElement(net_structure, 'Middle_Block')
        middle_layers_structure_num = ET.SubElement(middle_layers_structure, 'num')
        middle_layers_structure_num.text = str(middle_block['num'])
        for i in range(middle_block['num']):
            mdl_layer_str = 'Layer_%d' % i
            mdl_layer_i = ET.SubElement(middle_layers_structure, mdl_layer_str)
            create_layer_node(mdl_layer_i, {'units': middle_block['units'][i], 'activate': middle_block['activate'][i]})
        # 输出层
        output_layer_structure = ET.SubElement(net_structure, 'Output_Block')
        create_layer_node(output_layer_structure, {'units': output_block['dim'], 'activate': output_block['activate']})
        # 保存数据
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True, pretty_print=True)
    return file_path
