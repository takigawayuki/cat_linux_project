module test_top(
  input clk,
  input rstn,
  output wire                          eth_rst_n_0        , //以太网复位信号
  input  wire                          eth_rgmii_rxc_0    ,
  input  wire                          eth_rgmii_rx_ctl_0 ,
  input  wire [3:0]                    eth_rgmii_rxd_0    ,  
/*   input  wire [15:0]                   img_x1             ,
  input  wire [15:0]                   img_y1             ,
  input  wire [15:0]   			       img_x2             ,
  input  wire [15:0]   			       img_y2             ,     */                 
  output wire                          eth_rgmii_txc_0    ,
  output wire                          eth_rgmii_tx_ctl_0 ,
  output wire [3:0]                    eth_rgmii_txd_0    





   );
   //UDP通信
//开发板MAC地址 00-11-22-33-44-55
parameter  BOARD_MAC = 48'ha0_b1_c2_d3_e1_e1;     
//开发板IP地址 192.168.1.10     
parameter  BOARD_IP  = {8'd192,8'd168,8'd1,8'd11};
//目的MAC地址 ff_ff_ff_ff_ff_ff
parameter  DES_MAC   = 48'h3e_f9_49_84_9e_ee;
//parameter  DES_MAC   = 48'hff_ff_ff_ff_ff_ff;

//目的IP地址 192.168.1.102
parameter  DES_IP    = {8'd192,8'd168,8'd1,8'd22};
parameter	    v_disp_a = 180;    
parameter       h_disp_a = 320;
parameter  		video_h_pixel_0 = 11'd480 ;    
parameter 		video_v_pixel_0 = 11'd270 ;
wire pix_clk;
wire locked; 
 pll pll_inst1 (
  .clkin1(clk),        // input
  .pll_lock(locked),    // output
  .clkout0(pix_clk)       // output
); 
reg [15:0]rstn_1ms;
    always @(posedge pix_clk)
    begin
    	if(!locked)
    	    rstn_1ms <= 16'd0;
    	else
    	begin
    		if(rstn_1ms == 16'h2710)
    		    rstn_1ms <= rstn_1ms;
    		else
    		    rstn_1ms <= rstn_1ms + 1'b1;
    	end
    end
    
    assign rstn_out = (rstn_1ms == 16'h2710);
	assign eth_rst_n_0 = rstn_out;
/* wire rstn;
wire reset_n;
wire lock;
reg [1:0]Reset_reg=2'b0;
   always@(posedge pix_clk )
   Reset_reg<={Reset_reg[0],rst_board};
   assign reset_n=Reset_reg[1];
   reg [1:0]pll_lock_reg=2'b0;
   always@(posedge pix_clk )
   pll_lock_reg<={pll_lock_reg[0],pll_lock};
   assign lock=pll_lock_reg[1];
   assign rstn=(lock==1)?reset_n:1'd0; */
wire rec_pkt_done;
wire rec_en;
wire [31:0]rec_data;
wire [15:0]rec_byte_num;
wire eth0_rx_de;
wire eth0_rx_vs;
wire [15:0]eth0_rx_data;
eth_img_rec
//#(
//parameter integer PIXEL_WIDTH = 32                                   ,
//parameter integer VIDEO_LENGTH = 16'd960                             ,
//parameter integer VIDEO_HIGTH  = 16'd540                             
//)
eth0_img_rec(
.eth_rx_clk   (rgmii_clk_0  ),//input wire                         
.rstn         (rstn    ),//input wire                         
.udp_date_rcev(rec_data     ),//input wire [31: 0]   
.udp_date_en  (rec_en       ),//input wire                         
.img_data_en  (eth0_rx_de  ),//output reg                         
.img_data_vs  (eth0_rx_vs  ),//output reg                         
.img_data     (eth0_rx_data) //output reg [15: 0]   
 );
   parameter               X_WIDTH=12;
    parameter              Y_WIDTH=12;
   
//MODE_1080p
    parameter V_TOTAL = 12'd1125;
    parameter V_FP = 12'd4;
    parameter V_BP = 12'd36;
    parameter V_SYNC = 12'd5;
    parameter V_ACT = 12'd1080;
    parameter H_TOTAL = 12'd2200;
    parameter H_FP = 12'd88;
    parameter H_BP = 12'd148;
    parameter H_SYNC = 12'd44;
    parameter H_ACT = 12'd1920;
    parameter HV_OFFSET = 12'd0;	

/*   //480mode
	parameter X_WIDTH=4'd12;
    parameter Y_WIDTH=4'd12;
	parameter V_TOTAL = 12'd525;  // 场扫描周期 (480 + 10 + 2 + 33)
	parameter V_FP   = 12'd10;    // 场显示前沿
	parameter V_BP   = 12'd33;    // 场显示后沿
	parameter V_SYNC = 12'd2;     // 场同步宽度
	parameter V_ACT  = 12'd480;   // 场有效数据

	parameter H_TOTAL = 12'd800;  // 行扫描周期 (640 + 16 + 96 + 48)
	parameter H_FP   = 12'd16;    // 行显示前沿
	parameter H_BP   = 12'd48;    // 行显示后沿
	parameter H_SYNC = 12'd96;    // 行同步宽度
	parameter H_ACT  = 12'd640;   // 行有效数据
    parameter HV_OFFSET = 12'd0;  	 */
wire i_vs;
wire i_hs;
wire i_de;
wire [11:0]act_x;
sync_vg  #(
        .X_BITS               (  X_WIDTH              ), 
        .Y_BITS               (  Y_WIDTH              ),
        .V_TOTAL              (  V_TOTAL              ),//                        
        .V_FP                 (  V_FP                 ),//                        
        .V_BP                 (  V_BP                 ),//                        
        .V_SYNC               (  V_SYNC               ),//                        
        .V_ACT                (  V_ACT                ),//                        
        .H_TOTAL              (  H_TOTAL              ),//                        
        .H_FP                 (  H_FP                 ),//                        
        .H_BP                 (  H_BP                 ),//                        
        .H_SYNC               (  H_SYNC               ),//                        
        .H_ACT                (  H_ACT                ) //                        
 
    ) sync_vg_inst_2(
                  .clk(pix_clk),
				  .rstn(rstn), 
                  .vs_out(i_vs),
                  .hs_out(i_hs),
                  .de_out(i_de),
				  .x_act(act_x),
				  .y_act()
);
wire [7:0]r;
wire [7:0]g;
wire [7:0]b;
wire vs;
wire hs;
wire de;
pattern_vg # (
		
                .X_BITS(X_WIDTH),
                .Y_BITS(Y_WIDTH),
				.H_ACT(H_ACT),
				.V_ACT(V_ACT)

)pattern_vg_inst_2(                 
										
                                    .rstn(rstn), 
                                    .pix_clk(pix_clk),
									.act_x(act_x),
                                    .vs_in(i_vs), 
                                    .hs_in(i_hs), 
                                    .de_in(i_de),
			
									.vs_out(vs), 
									.hs_out(hs), 
									.de_out(de),
									.r_out(r), 
									.g_out(g), 
									.b_out(b)
);

wire [31:0]rgb;
assign     rgb        =    {r,2'b0,g,2'b0,b,4'b0};//{r,g,b}

////////////////////////////////////////////////////////////////////
wire [31:0]data_o;
 video_zoom #(
     .PIXEL_WIDTH (32)          ,
     .VIDEO_LENGTH (H_ACT)    ,
     .VIDEO_HIGTH (V_ACT)   	
)video_zoom_inst1(
                                    .clk(pix_clk)            ,
                                    .rstn(rstn)           ,
                                    .vs_in(vs)          /* synthesis PAP_MARK_DEBUG="1" */,
                                    .hs_in(hs)         ,
                                    .de_in(de)          /* synthesis PAP_MARK_DEBUG="1" */,
									.video_data_in(rgb)  ,
									.de_out(de_o)          /* synthesis PAP_MARK_DEBUG="1" */,
									.video_data_out(data_o)    /* synthesis PAP_MARK_DEBUG="1" */
   );
   
wire [15:0] video_frame_data;  
 video_tailor_2x2
#(
	  .v_disp_a(v_disp_a),    
      .h_disp_a(h_disp_a),
  	  .video_h_pixel_0(video_h_pixel_0) ,    
	  .video_v_pixel_0(video_v_pixel_0) 




)video_tailor_2x2_inst1(
                   .rst_n(rstn)            ,  //复位信号                     
                              
				   .video_pclk(pix_clk)         ,  //video数据像素时钟
				   .video_vsync(vs)        ,  //video 场同步信号（负极性）
				   .video_hs(~de_o)         ,  //video行同步信号（负极性）
				   .video_data({data_o[31 : 27],data_o[21 : 16],data_o[11 :  7]})         , 
				   .video_data_valid(de_o)   ,    
                                 
				   .video_frame_valid(video_frame_valid) ,  //数据有效使能信号
				   .video_frame_data(video_frame_data)     //有效数据        
    );  
//wire rgmii_clk_0;
wire tx_req;
wire udp_tx_done;
wire tx_start_en;
wire [31:0]tx_data;
wire [15:0]tx_byte_num;
 eth_img_pkt eth0_img_pkt(    
    .rst_n              (rstn            ), //input                    
    ////图像相关信号              
    .cam_pclk           (pix_clk         ), //input  图像时钟             
    .img_vsync          (vs            ), //input  帧同步               
    .img_data_en        (video_frame_valid            ), //input  de               
    .img_data           (video_frame_data       ), //input  [15:0]   //vesa_debug_data //eth0_img_data
    .transfer_flag      (1               ), //input    
/* 	.img_x1             (  16'h0104      ),
	.img_y1             (  16'h00A5      ),
	.img_x2             (  16'h01CB      ),
	.img_y2             (  16'h013A      ), */
    ////以太网相关信号
    .eth_tx_clk         (rgmii_clk_0     ), //input                          
    .udp_tx_req         (tx_req          ), //input                
    .udp_tx_done        (udp_tx_done     ), //input                
    .udp_tx_start_en    (tx_start_en     ), //output  reg          
    .udp_tx_data        (tx_data         ), //output       [31:0]  
    .udp_tx_byte_num    (tx_byte_num     )  //output  reg  [15:0]  
    ); 


wire mac_rx_data_valid_0;
wire  [7:0]mac_rx_data_0;
wire mac_tx_data_valid_0;
wire  [7:0]mac_tx_data_0;

udp_top                                             
   #(
    .BOARD_MAC     (BOARD_MAC),      //参数例化
    .BOARD_IP      (BOARD_IP ),
    .DES_MAC       (DES_MAC  ),
    .DES_IP        (DES_IP   )
    )
u_udp(
    .rst_n         (rstn   ),  //input       复位信号，低电平有效            
    //GMII接口                                
    .gmii_rx_clk   (rgmii_clk_0         ),  //input       GMII接收数据时钟                    
    .gmii_rx_dv    (mac_rx_data_valid_0 ),  //input       GMII输入数据有效信号                
    .gmii_rxd      (mac_rx_data_0       ),  //input [7:0] GMII输入数据                              
    .gmii_tx_clk   (rgmii_clk_0         ),  //input       GMII发送数据时钟            
    .gmii_tx_en    (mac_tx_data_valid_0 ),  //output      GMII输出数据有效信号                  
    .gmii_txd      (mac_tx_data_0       ),  //output[7:0] GMII输出数据              
    //用户接口                                  
    .rec_pkt_done  (rec_pkt_done        ),  //output      以太网单包数据接收完成信号          
    .rec_en        (rec_en              ),  //output      以太网接收的数据使能信号            
    .rec_data      (rec_data            ),  //output[31:0]以太网接收的数据                    
    .rec_byte_num  (rec_byte_num        ),  //output[15:0]以太网接收的有效字节数 单位:byte  
    
    .tx_start_en   (tx_start_en         ),  //input       以太网开始发送信号                  
    .tx_data       (tx_data             ),  //input [31:0]以太网待发送数据                    
    .tx_byte_num   (tx_byte_num         ),  //input [15:0]以太网发送的有效字节数 单位:byte   
    .des_mac       (DES_MAC             ),  //input [47:0]发送的目标MAC地址            
    .des_ip        (DES_IP              ),  //input [31:0]发送的目标IP地址              
    .tx_done       (udp_tx_done         ),  //output      以太网发送完成信号                  
    .tx_req        (tx_req              )   //output      读数据请求信号                      
    ); 
//ETH0_GMII_RGMII
gmii_to_rgmii eth0_gmii_to_rgmii(
   .rgmii_clk             (rgmii_clk_0       ),    // output GMII时钟，供数据使用      
   .rst                   (rstn         ),    // input        
    //mac输入的数据由gmii转化为rgmii，时钟为rgmii_clk
   .mac_tx_data_valid     (mac_tx_data_valid_0),    // input        
   .mac_tx_data           (mac_tx_data_0      ),    // input [7:0]  
    //eth输入的数据由rgmii转化为gmii，时钟为rgmii_clk
   .mac_rx_error          (mac_rx_error_0     ),    //output reg       
   .mac_rx_data_valid     (mac_rx_data_valid_0),    //output reg       
   .mac_rx_data           (mac_rx_data_0      ),    //output reg [7:0] 
   //eth接收                
   .rgmii_rxc             (eth_rgmii_rxc_0    ),    //input        
   .rgmii_rx_ctl          (eth_rgmii_rx_ctl_0 ),    //input        
   .rgmii_rxd             (eth_rgmii_rxd_0    ),    //input [3:0]  
   //eth发送                                    
   .rgmii_txc             (eth_rgmii_txc_0    ),    //output       
   .rgmii_tx_ctl          (eth_rgmii_tx_ctl_0 ),    //output       
   .rgmii_txd             (eth_rgmii_txd_0    )     //output [3:0] 
);

	
endmodule