module eth_img_pkt(
    input                 rst_n          ,   //��λ�źţ��͵�ƽ��Ч
    //ͼ������ź�
    input                 cam_pclk       /* synthesis PAP_MARK_DEBUG="1" */,   //����ʱ��
    input                 img_vsync      /* synthesis PAP_MARK_DEBUG="1" */,   //֡ͬ���ź�
    input                 img_data_en    /* synthesis PAP_MARK_DEBUG="1" */,   //������Чʹ���ź�
    input        [15:0]   img_data       ,   //��Ч���� 
    
    input                 transfer_flag  ,   //ͼ��ʼ�����־,1:��ʼ���� 0:ֹͣ����
    //��̫������ź� 
    input                 eth_tx_clk     ,   //��̫������ʱ��
    input                 udp_tx_req     /* synthesis PAP_MARK_DEBUG="1" */,   //udp�������������ź�
    input                 udp_tx_done    /* synthesis PAP_MARK_DEBUG="1" */,   //udp������������ź�                               
    output  reg           udp_tx_start_en,   //udp��ʼ�����ź�
    output       [31:0]   udp_tx_data    /* synthesis PAP_MARK_DEBUG="1" */,   //udp���͵�����
    output  reg  [15:0]   udp_tx_byte_num    //udp�������͵���Ч�ֽ���
    );    
    
//parameter define
parameter  CMOS_H_PIXEL = 16'd480;  //ͼ��ˮƽ����ֱ���
parameter  CMOS_V_PIXEL = 16'd270;  //ͼ��ֱ����ֱ���
parameter  UDP_DATA_SIZE = 16'd960; //UDP���ݳ��ȣ��������ײ���
parameter  ETH_TRAN_DELAY = 16'd6375; //֡����ӳ�
//ͼ��֡ͷ,���ڱ�־һ֡���ݵĿ�ʼ
parameter  IMG_FRAME_HEAD = {32'hf0_5a_a5_0f};

reg             img_vsync_d0    ;  //֡��Ч�źŴ���
reg             img_vsync_d1    ;  //֡��Ч�źŴ���
reg             neg_vsync_d0    ;  //֡��Ч�ź��½��ش���
reg             neg_vsync_d1    ;
reg             img_de_d0       ;  //DE��־λ����
reg             img_de_d1       ;  //DE��־λ ����     
reg             eth_delay_d0    ;
reg             eth_delay_d1    ;                          


reg             wr_sw           ;  //����λƴ�ӵı�־
reg    [15:0]   img_data_d0     ;  //��Чͼ�����ݴ���
reg             wr_fifo_en      /* synthesis PAP_MARK_DEBUG="1" */;  //дfifoʹ��
reg    [31:0]   wr_fifo_data    /* synthesis PAP_MARK_DEBUG="1" */;  //дfifo����

reg             img_vsync_txc_d0;  //��̫������ʱ������,֡��Ч�źŴ���
reg             img_vsync_txc_d1;  //��̫������ʱ������,֡��Ч�źŴ���
reg             tx_busy_flag    ;  //����æ�źű�־
reg    [15 : 0] img_de_cnt      /* synthesis PAP_MARK_DEBUG="1" */;//de�½��ؼ���
reg    [31 : 0] img_pkt_cnt    /* synthesis PAP_MARK_DEBUG="1" */;
reg    [15 : 0]       eth_delay_cnt  /* synthesis PAP_MARK_DEBUG="1" */;    //delay������  
reg                  eth_delay_start/* synthesis PAP_MARK_DEBUG="1" */;
reg                  eth_delay_done /* synthesis PAP_MARK_DEBUG="1" */;  
reg neg_vsync_r_flg_d0;
reg neg_vsync_r_flg_d1;
reg [31:0]pos_delay_cnt/* synthesis PAP_MARK_DEBUG="1" */;
reg [31:0]neg_delay_cnt/* synthesis PAP_MARK_DEBUG="1" */;
reg pos_reach;
reg neg_reach;
reg pos_vsync_r_flg/* synthesis PAP_MARK_DEBUG="1" */;
reg neg_vsync_r_flg /* synthesis PAP_MARK_DEBUG="1" */;
/* reg f_vsync_txc_d0;
reg f_vsync_txc_d1; */
//wire define                   
wire            pos_vsync       /* synthesis PAP_MARK_DEBUG="1" */;  //֡��Ч�ź�������
wire            neg_vsync       /* synthesis PAP_MARK_DEBUG="1" */;  //֡��Ч�ź��½���
wire            neg_vsynt_txc   ;  //��̫��ʱ��֡�½���ɲ���źŵ������ؼ�⣨α֡�½��ؼ�⣩
wire            pos_de          ;  //������Ч�ź�������
wire            pos_eth_delay   ;  //udp
wire   [15:0]    fifo_rdusedw    /* synthesis PAP_MARK_DEBUG="1" */;  //��ǰFIFO����ĸ���
/* wire 			pos_vsynt_txc;	 */	

//*****************************************************
//**                    main code
//*****************************************************

//�źŲ���
assign neg_vsync = img_vsync_d1 & (~img_vsync_d0);
assign pos_vsync = ~img_vsync_d1 & img_vsync_d0;
assign neg_vsynt_txc = img_vsync_txc_d0 & (~img_vsync_txc_d1);
assign neg_de    = (~img_de_d0) & img_de_d1;
assign pos_eth_delay = eth_delay_d0 & (~eth_delay_d1);
/* assign pos_vsynt_txc = f_vsync_txc_d0 & (~f_vsync_txc_d1);  */
//��img_vsync�ź���ʱ����ʱ������,���ڲ���
always @(posedge cam_pclk or negedge rst_n) begin
    if(!rst_n) begin
        img_vsync_d0 <= 1'b0;
        img_vsync_d1 <= 1'b0;
        img_de_d0    <= 1'b0;
        img_de_d1    <= 1'b0;
        eth_delay_d0 <=  'd0;//������ʱ�����¶�udp��������źŲ���
        eth_delay_d1 <=  'd0;
		neg_vsync_r_flg_d0 <= 1'b0;
        neg_vsync_r_flg_d1 <= 1'b0;
    end
    else begin
        img_vsync_d0 <= img_vsync;
        img_vsync_d1 <= img_vsync_d0;
        img_de_d0 <= img_data_en;
        img_de_d1 <= img_de_d0;
        eth_delay_d0 <= eth_delay_start;//������ʱ�����¶�udp��������źŲ���
        eth_delay_d1 <= eth_delay_d0;
		neg_vsync_r_flg_d0 <= neg_vsync_r_flg;
        neg_vsync_r_flg_d1 <= neg_vsync_r_flg_d0;
    end
end

//֡��Ч�ź��������ӳ�
always @(posedge cam_pclk or negedge rst_n) begin
    if(!rst_n) begin
		pos_delay_cnt <= 'd0;
		pos_reach <= 1'd0;
		pos_vsync_r_flg <= 1'd0;
	end
	else begin	
		if(pos_vsync)begin 
			pos_delay_cnt <= 'd0;
			pos_reach <= 1'd0;
			pos_vsync_r_flg <= 1'd0;
		end
	else begin 
		if(!pos_reach)begin 
		if(pos_delay_cnt == 'd300000)begin 
			pos_delay_cnt <= 'd0;
			pos_reach <= 1'd1;
			pos_vsync_r_flg <= 1'd1;
		end
		else begin 
		pos_delay_cnt <= pos_delay_cnt+'d1;
		pos_vsync_r_flg <= 1'd0;
		end
		end
		else begin  
		pos_delay_cnt <= 'd0;
		pos_vsync_r_flg <= 1'd0;
		end
	end	
	end
end

always @(posedge cam_pclk or negedge rst_n) begin
    if(!rst_n) begin
		neg_delay_cnt <= 'd0;
		neg_reach <= 1'd0;
		neg_vsync_r_flg <= 1'd0;
	end
	else begin	
		if(neg_vsync)begin 
			neg_delay_cnt <= 'd0;
			neg_reach <= 1'd0;
			neg_vsync_r_flg <= 1'd0;
		end
	else begin 
		if(!neg_reach)begin 
		if(neg_delay_cnt == 'd291560)begin 
			neg_delay_cnt <= 'd0;
			neg_reach <= 1'd1;
			neg_vsync_r_flg <= 1'd1;
		end
		else begin 
		neg_delay_cnt <= neg_delay_cnt+'d1;
		neg_vsync_r_flg <= 1'd0;
		end
		end
		else begin  
		neg_delay_cnt <= 'd0;
		neg_vsync_r_flg <= 1'd0;
		end
	end	
	end
end
//�Ĵ�neg_vsync�ź�
always @(posedge cam_pclk or negedge rst_n) begin
    if(!rst_n) begin
        neg_vsync_d0 <= 1'b0;
        neg_vsync_d1 <= 1'b0;
    end    
    else begin 
        neg_vsync_d0 <= neg_vsync;
        neg_vsync_d1 <= neg_vsync_d0;
    end
end    

//��wr_sw��img_data_d0�źŸ�ֵ,����λƴ��
always @(posedge cam_pclk or negedge rst_n) begin
    if(!rst_n) begin
        wr_sw <= 1'b0;
        img_data_d0 <= 1'b0;
    end
     else if(neg_vsync)
        wr_sw <= 1'b0;
    else if(img_data_en) begin
        wr_sw <= ~wr_sw;
        img_data_d0 <= img_data;
    end    
end 
//////////
reg cnt;//��de�½��ؼ������ϲ��ļ��ü�����һ��de���л�仯����
always @(posedge cam_pclk or negedge rst_n) begin
    if(!rst_n) begin
        cnt <= 1'b0;
    end
     else if(neg_de)
        cnt <= cnt+'d1;
    else
		cnt <= cnt;   
end 
//��֡ͷ��ͼ������д��FIFO
always @(posedge cam_pclk or negedge rst_n) begin
    if(!rst_n || pos_vsync_r_flg) begin
        wr_fifo_en <= 1'b0;
        wr_fifo_data <= 32'b0;
        img_de_cnt <= 32'd0;
        img_pkt_cnt <= 'd1;
    end
    else begin
        if(neg_vsync_r_flg) begin
            wr_fifo_en <= 1'b1;
            wr_fifo_data <= img_pkt_cnt;                  //UDP����־λ
            img_pkt_cnt <= img_pkt_cnt + 'd1;
        end
        else if(neg_vsync_r_flg_d0) begin
            wr_fifo_en <= 1'b1;
            wr_fifo_data <= IMG_FRAME_HEAD;               //֡ͷ
        end
        else if(neg_vsync_r_flg_d1) begin
            wr_fifo_en <= 1'b1;
            wr_fifo_data <= {CMOS_H_PIXEL,CMOS_V_PIXEL};  //ˮƽ�ʹ�ֱ����ֱ���
        end
        else if(img_data_en && wr_sw) begin
            wr_fifo_en <= 1'b1;
            img_de_cnt <= img_de_cnt + 'd1;
            wr_fifo_data <= {img_data_d0,img_data};       //ͼ������λƴ��,16λת32λ
        end
/*         else if(img_de_cnt == 240) begin
            wr_fifo_en <= 1'b1;
            wr_fifo_data <= img_pkt_cnt;
            img_pkt_cnt <= img_pkt_cnt + 'd1;
            
        end */
        else if(neg_de&&(cnt==1)) begin  
            wr_fifo_en <= 1'b1;
            wr_fifo_data <= img_pkt_cnt;           //de������д�����
            img_pkt_cnt <= img_pkt_cnt + 'd1;
            img_de_cnt <= 'd0;
        end
        else begin
            wr_fifo_en <= 1'b0;
            wr_fifo_data <= 32'b0;        
        end
    end
end


always @(posedge eth_tx_clk or negedge rst_n) begin
    if(!rst_n) begin
        img_vsync_txc_d0 <= 1'b0;
        img_vsync_txc_d1 <= 1'b0;
    end
    else begin
        img_vsync_txc_d0 <= neg_reach;
        img_vsync_txc_d1 <= img_vsync_txc_d0;
    end
end

/* always @(posedge eth_tx_clk or negedge rst_n) begin
    if(!rst_n) begin
        f_vsync_txc_d0 <= 1'b0;
        f_vsync_txc_d1 <= 1'b0;
    end
    else begin
        f_vsync_txc_d0 <= pos_reach;
        f_vsync_txc_d1 <= f_vsync_txc_d0;
    end
end */
//������̫�����͵��ֽ���
always @(posedge eth_tx_clk or negedge rst_n) begin
    if(!rst_n)
        udp_tx_byte_num <= 1'b0;
    else if(neg_vsynt_txc)//vs���һ�����ݰ�����ͼ�����ݳ���+��ʼ��־λ����8Byte
        //udp_tx_byte_num <= {CMOS_H_PIXEL,1'b0} + 16'd8;
        udp_tx_byte_num <= UDP_DATA_SIZE + 16'd12;//16'd960 + 16'd8;
    else if(udp_tx_done)//������������    
        //udp_tx_byte_num <= {CMOS_H_PIXEL,1'b0};
        udp_tx_byte_num <= UDP_DATA_SIZE + 16'd4;//16'd960;
end

//������̫�����Ϳ�ʼ�ź�
always @(posedge eth_tx_clk or negedge rst_n) begin
    if(!rst_n) begin
        udp_tx_start_en <= 1'b0;
        tx_busy_flag <= 1'b0;
    end
    //��λ��δ����"��ʼ"����ʱ,��̫��������ͼ������
    else if(transfer_flag == 1'b0) begin
        udp_tx_start_en <= 1'b0;
        tx_busy_flag <= 1'b0;        
    end
    else begin
        udp_tx_start_en <= 1'b0;
        //��FIFO�еĸ���������Ҫ���͵��ֽ���ʱ ˮλ/4(32bit->8bit)
        if(tx_busy_flag == 1'b0 && fifo_rdusedw >= udp_tx_byte_num[15:2]&&(udp_tx_byte_num>0)) begin
            udp_tx_start_en <= 1'b1;                     //��ʼ���Ʒ���һ������
            tx_busy_flag <= 1'b1;
        end
        else if(eth_delay_done|| neg_vsynt_txc) 
            tx_busy_flag <= 1'b0;
    end
end
//��С֡�����������96bits�ļ��,ǧ����̫��Ҫ��96ns��125Mhz->8ns����12��ʱ�����ڣ������500+��
always @(posedge eth_tx_clk or negedge rst_n) begin
    if(!rst_n) begin
        eth_delay_cnt <= 'd0;
        eth_delay_start <= 'd0;
        eth_delay_done <= 'd0;
    end
    else begin
        eth_delay_done <= 'd0;
        if(udp_tx_done) begin
            eth_delay_start <= 'd1;
        end 
        if(eth_delay_start) begin
            eth_delay_cnt <= eth_delay_cnt + 'd1;
        end
        if(eth_delay_cnt >= ETH_TRAN_DELAY) begin
            eth_delay_done  <= 'd1;
            eth_delay_start <= 'd0;
            eth_delay_cnt   <= 'd0;
        end
    end
end
reg    [10:0]   udp_pkt_cnt     /* synthesis PAP_MARK_DEBUG="1" */;  //udp������
always @(posedge eth_tx_clk or negedge rst_n) begin
    if(!rst_n || neg_vsynt_txc) begin
        udp_pkt_cnt <= 'd0;
    end    
    else if(udp_tx_done)begin
        udp_pkt_cnt <= udp_pkt_cnt + 'd1;
    end
    else begin
        udp_pkt_cnt <= udp_pkt_cnt;
    end
end
//�첽FIFO

eth_pkt_fifo u_eth_pkt_fifo (
  .wr_clk           (cam_pclk                    ) ,    // input
  .wr_rst           (pos_vsync_r_flg | (~transfer_flag)) ,    // input
  .wr_en            (wr_fifo_en                  ) ,    // input
  .wr_data          (wr_fifo_data                ) ,    // input [31:0]
  .wr_full          (                            ) ,    // output
  .wr_water_level   (  				             ) ,    // output [15:0]
  .almost_full      (                            ) ,    // output
  .rd_clk           (eth_tx_clk                  ) ,    // input
  .rd_rst           (pos_vsync_r_flg | (~transfer_flag)) ,    // input
  .rd_en            (udp_tx_req                  ) ,    // input
  .rd_data          (udp_tx_data                 ) ,    // output [31:0]
  .rd_empty         (                            ) ,    // output
  .rd_water_level   (fifo_rdusedw                ) ,    // output [15:0]
  .almost_empty     (                            )      // output
);
//eth_pkt_fifo u_eth_pkt_fifo (
//  .wr_clk(wr_clk),                    // input
//  .wr_rst(wr_rst),                    // input
//  .wr_en(wr_en),                      // input
//  .wr_data(wr_data),                  // input [31:0]
//  .wr_full(wr_full),                  // output
//  .wr_water_level(wr_water_level),    // output [9:0]
//  .almost_full(almost_full),          // output
//  .rd_clk(rd_clk),                    // input
//  .rd_rst(rd_rst),                    // input
//  .rd_en(rd_en),                      // input
//  .rd_data(rd_data),                  // output [31:0]
//  .rd_empty(rd_empty),                // output
//  .rd_water_level(rd_water_level),    // output [9:0]
//  .almost_empty(almost_empty)         // output
//);

endmodule