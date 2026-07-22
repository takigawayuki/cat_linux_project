`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:Meyesemi 
// Engineer: Will
// 
// Create Date: 2023-01-29 20:31  
// Design Name:  
// Module Name: 
// Project Name: 
// Target Devices: Pango
// Tool Versions: 
// Description: 
//      
// Dependencies: 
// 
// Revision:
// Revision 1.0 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
`define UD #1

module hdmi_loop(
    input wire        sys_clk,     // input system clock 50MHz
    
    output            rstn_out,
    output            iic_scl,
    inout             iic_sda, 
    output            iic_tx_scl,
    inout             iic_tx_sda, 
    input             pixclk_in,                            
    input             vs_in, 
    input             hs_in, 
    input             de_in,
    input     [7:0]   r_in, 
    input     [7:0]   g_in, 
    input     [7:0]   b_in,  

    output               pixclk_out,                            
    output reg           vs_out, 
    output reg           hs_out, 
    output reg           de_out,
    output reg    [7:0]  r_out, 
    output reg    [7:0]  g_out, 
    output reg    [7:0]  b_out,
    output               led_int,
  output wire                          eth_rst_n_0        , //以太网复位信号
  input  wire                          eth_rgmii_rxc_0    ,
  input  wire                          eth_rgmii_rx_ctl_0 ,
  input  wire [3:0]                    eth_rgmii_rxd_0    ,  
               
  output wire                          eth_rgmii_txc_0    ,
  output wire                          eth_rgmii_tx_ctl_0 ,
  output wire [3:0]                    eth_rgmii_txd_0    
);

    reg  [15:0] rstn_1ms;
    wire        cfg_clk;
    wire        locked;
    wire        init_over;

    PLL u_pll (
        .clkin1   (sys_clk),
        .pll_lock (locked),
        .clkout0  (cfg_clk)
    );

    wire unused_clahe_pix_clk;
    wire unused_clahe_cfg_clk;
    wire clahe_clk;
    wire clahe_locked;

    clahe_pll u_clahe_pll (
        .clkin1   (sys_clk),
        .clkout0  (unused_clahe_pix_clk),
        .clkout1  (unused_clahe_cfg_clk),
        .clkout2  (clahe_clk),
        .pll_lock (clahe_locked)
    );

    ms72xx_ctl ms72xx_ctl(
        .clk         (cfg_clk),
        .rst_n       (rstn_out),
        .init_over   (init_over),
        .iic_tx_scl  (iic_tx_scl),
        .iic_tx_sda  (iic_tx_sda),
        .iic_scl     (iic_scl),
        .iic_sda     (iic_sda)
    );

    assign led_int = init_over;

    always @(posedge cfg_clk)
    begin
        if(!locked)
            rstn_1ms <= 16'd0;
        else if(rstn_1ms != 16'h2710)
            rstn_1ms <= rstn_1ms + 1'b1;
    end

    assign rstn_out = (rstn_1ms == 16'h2710);
    assign pixclk_out = pixclk_in;

    wire processing_rst_n = rstn_out && clahe_locked;
	assign eth_rst_n_0 = processing_rst_n;
    // Synchronize transmitter initialization into the incoming pixel domain.
    // Start processing on a frame boundary so the first enabled frame is whole.
    reg init_sync1;
    reg init_sync2;
    reg vs_in_d;
    reg stream_enable;

    always @(posedge pixclk_in or negedge processing_rst_n) begin
        if (!processing_rst_n) begin
            init_sync1  <= 1'b0;
            init_sync2  <= 1'b0;
            vs_in_d     <= 1'b0;
            stream_enable <= 1'b0;
        end else begin
            init_sync1 <= init_over;
            init_sync2 <= init_sync1;
            vs_in_d    <= vs_in;

            if (!init_sync2)
                stream_enable <= 1'b0;
            else if (!stream_enable && vs_in && !vs_in_d)
                stream_enable <= 1'b1;
        end
    end

    wire [4:0] rgb565_r = r_in[7:3];
    wire [5:0] rgb565_g = g_in[7:2];
    wire [4:0] rgb565_b = b_in[7:3];

    wire [7:0] y_in;
    wire [7:0] cb_in;
    wire [7:0] cr_in;
    wire       vs_yuv;
    wire       hs_yuv;
    wire       de_yuv;

    RGB2YCbCr u_RGB2YCbCr (
        .clk       (pixclk_in),
        .rst_n     (processing_rst_n),
        .vsync_in  (vs_in),
        .hsync_in  (hs_in),
        .de_in     (de_in && stream_enable),
        .red       (rgb565_r),
        .green     (rgb565_g),
        .blue      (rgb565_b),
        .vsync_out (vs_yuv),
        .hsync_out (hs_yuv),
        .de_out    (de_yuv),
        .y         (y_in),
        .cb        (cb_in),
        .cr        (cr_in)
    );

    wire [7:0] y_clahe;
    wire [7:0] cb_clahe;
    wire [7:0] cr_clahe;
    wire       vs_clahe;
    wire       hs_clahe;
    wire       de_clahe;

    clahe_hdmi_bridge #(
        .VIDEO_DELAY (64)
    ) u_clahe_hdmi_bridge (
        .pix_clk    (pixclk_in),
        .core_clk   (clahe_clk),
        .rst_n      (processing_rst_n),
        .vs_in      (vs_yuv),
        .hs_in      (hs_yuv),
        .de_in      (de_yuv),
        .y_in       (y_in),
        .cb_in      (cb_in),
        .cr_in      (cr_in),
        .clip_limit (16'd400),
        .enhance_strength(2'd2),
        .vs_out     (vs_clahe),
        .hs_out     (hs_clahe),
        .de_out     (de_clahe),
        .y_out      (y_clahe),
        .cb_out     (cb_clahe),
        .cr_out     (cr_clahe)
    );

    wire [4:0] rgb_out_r;
    wire [5:0] rgb_out_g;
    wire [4:0] rgb_out_b;
    wire       vs_rgb;
    wire       hs_rgb;
    wire       de_rgb;

    YCbCr2RGB u_YCbCr2RGB (
        .clk       (pixclk_in),
        .rst_n     (processing_rst_n),
        .vsync_in  (vs_clahe),
        .hsync_in  (hs_clahe),
        .de_in     (de_clahe),
        .y         (y_clahe),
        .cb        (cb_clahe),
        .cr        (cr_clahe),
        .vsync_out (vs_rgb),
        .hsync_out (hs_rgb),
        .de_out    (de_rgb),
        .red       (rgb_out_r),
        .green     (rgb_out_g),
        .blue      (rgb_out_b)
    );

    always @(posedge pixclk_in or negedge processing_rst_n) begin
        if (!processing_rst_n) begin
            vs_out <= 1'b0;
            hs_out <= 1'b0;
            de_out <= 1'b0;
            r_out  <= 8'b0;
            g_out  <= 8'b0;
            b_out  <= 8'b0;
        end else begin
            vs_out <= vs_rgb;
            hs_out <= hs_rgb;
            de_out <= de_rgb;
            r_out  <= {rgb_out_r, rgb_out_r[4:2]};
            g_out  <= {rgb_out_g, rgb_out_g[5:4]};
            b_out  <= {rgb_out_b, rgb_out_b[4:2]};
        end
    end
////////////////////////

wire         video_enhance_vs_out;
wire         video_enhance_hs_out;
wire         video_enhance_de_out;
wire [7 : 0] video_enhance_r_out;
wire [7 : 0] video_enhance_g_out;
wire [7 : 0] video_enhance_b_out;

wire [7  : 0]    video_enhance_lightdown_num;
wire             video_enhance_lightdown_sw ;
wire [7  : 0]    video_enhance_darkup_num   ;
wire             video_enhance_darkup_sw    ;

video_enhance u_video_enhance(
.pix_clk(pixclk_in),//input  wire            
.vs_in  (vs_out),//input  wire            
.hs_in  (hs_out),//input  wire            
.de_in  (de_out),//input  wire         zoom_de_out              
.r_in   (r_out),//input  wire [7 : 0] zoom_data_out[31 : 24]   
.g_in   (g_out),//input  wire [7 : 0] zoom_data_out[21 : 14]   
.b_in   (b_out),//input  wire [7 : 0] zoom_data_out[11 :  4]
   
.vs_out (video_enhance_vs_out  ),//output wire                               
.hs_out (video_enhance_hs_out  ),//output wire            
.de_out (video_enhance_de_out  ),//output wire            
.r_out  (video_enhance_r_out   ),//output wire [7 : 0]    
.g_out  (video_enhance_g_out   ),//output wire [7 : 0]    
.b_out  (video_enhance_b_out   ), //output wire [7 : 0]    
.video_enhance_lightdown_num (7'd111),//input wire [7 : 0]            
.video_enhance_lightdown_sw  (7'd111 ),//input wire                    
.video_enhance_darkup_num    (7'd111   ),//input wire [7 : 0]            
.video_enhance_darkup_sw     (7'd111   )//input wire                            
   );	
///////////////////	
// pcie_dma_ctrl u_pcie_dam_ctrl(
   // .clk                (pclk_div2 ), //input wire   
   // .pix_clk_out        (pixclk_in),          
   // .rstn               (processing_rst_n), //input              
    
   // .axis_master_tvalid (axis_master_tvalid), //input wire                
   // .axis_master_tready (axis_master_tready), //output wire               
   // .axis_master_tdata  (axis_master_tdata), //input wire    [127:0]     
   // .axis_master_tkeep  (axis_master_tkeep), //input wire    [3:0]       
   // .axis_master_tlast  (axis_master_tlast), //input wire                
   // .axis_master_tuser  (axis_master_tuser), //input wire    [7:0]       
 
   // .ep_bus_num         (cfg_pbus_num), //input  [7 : 0]         
   // .ep_dev_num         (cfg_pbus_dev_num), //input  [4 : 0] 
        
   // .AXIS_S_TREADY      (axis_slave2_tready ), //input                  
   // .AXIS_S_TVALID      (axis_slave2_tvalid ), //output                 
   // .AXIS_S_TDATA       (axis_slave2_tdata  ), //output [127:0]         
   // .AXIS_S_TLAST       (axis_slave2_tlast  ), //output                 
   // .AXIS_S_TUSER       (axis_slave2_tuser  ), //output 
   // .hdmi_data_in       (pcie_data_out      ), // input 32bits
   // .vs_in              (video_enhance_vs_out             ),                   
   // .de_in              (video_enhance_de_out           ) ,
   // .video_enhance_lightdown_num (video_enhance_lightdown_num),// output reg [7 : 0]        
   // .video_enhance_lightdown_sw  (video_enhance_lightdown_sw ),// output reg                
   // .video_enhance_darkup_num    (video_enhance_darkup_num   ),// output reg [7 : 0]        
   // .video_enhance_darkup_sw     (video_enhance_darkup_sw    ) // output reg            
   // );
 // wire             pclk;
 // wire             pclk_div2;
 // wire             pcie_ref_clk;
 // wire             axis_master_tvalid;
 // wire             axis_master_tready;
 // wire    [127:0]  axis_master_tdata synthesis PAP_MARK_DEBUG="1";
 // wire    [3:0]    axis_master_tkeep;
 // wire             axis_master_tlast;
 // wire    [7:0]    axis_master_tuser;
 // wire             axis_slave0_tvalid;
 // wire             axis_slave0_tlast;
 // wire             axis_slave0_tuser;
 // wire    [127:0]  axis_slave0_tdata;
 // wire             axis_slave1_tvalid;
 // wire             axis_slave1_tlast;
 // wire             axis_slave1_tuser;
 // wire    [127:0]  axis_slave1_tdata;
 // wire    [15: 0] pcie_data_out synthesis PAP_MARK_DEBUG="1";
//assign    pcie_data_out =  r_de_out?{r_r_out,r_g_out,r_b_out,'hdd} : 'd0;
// assign    pcie_data_out =  video_enhance_de_out?{video_enhance_r_out[7:3],video_enhance_g_out[7:2],video_enhance_b_out[7:3]} : 'd0;

//----------------------------------------------------------rst debounce ----------------------------------------------------------
//ASYNC RST  define IPSL_PCIE_SPEEDUP_SIM when simulation
// hsst_rst_cross_sync_v1_0 #(
    // `ifdef IPSL_PCIE_SPEEDUP_SIM
    // .RST_CNTR_VALUE     (16'h10             )
    // `else
    // .RST_CNTR_VALUE     (16'hC000           )
    // `endif
// )
// u_refclk_buttonrstn_debounce(
    // .clk                (pcie_ref_clk            ),
    // .rstn_in            (rst_board       ),
    // .rstn_out           (sync_button_rst_n  )
// );

// hsst_rst_cross_sync_v1_0 #(
    // `ifdef IPSL_PCIE_SPEEDUP_SIM
    // .RST_CNTR_VALUE     (16'h10             )
    // `else
    // .RST_CNTR_VALUE     (16'hC000           )
    // `endif
// )
// u_refclk_perstn_debounce(
    // .clk                (pcie_ref_clk            ),
    // .rstn_in            (pcie_perst_n            ),
    // .rstn_out           (sync_perst_n       )
// );

// ipsl_pcie_sync_v1_0  u_ref_core_rstn_sync    (
    // .clk                (pcie_ref_clk            ),
    // .rst_n              (core_rst_n         ),
    // .sig_async          (1'b1               ),
    // .sig_synced         (ref_core_rst_n     )
// );

// ipsl_pcie_sync_v1_0  u_pclk_core_rstn_sync   (
    // .clk                (pclk               ),
    // .rst_n              (core_rst_n         ),
    // .sig_async          (1'b1               ),
    // .sig_synced         (s_pclk_rstn        )
// );

// ipsl_pcie_sync_v1_0  u_pclk_div2_core_rstn_sync   (
    // .clk                (pclk_div2          ),
    // .rst_n              (core_rst_n         ),
    // .sig_async          (1'b1               ),
    // .sig_synced         (s_pclk_div2_rstn   )
// );
//axis slave 2 interface
// wire            axis_slave2_tready      ;
// wire            axis_slave2_tvalid      ;
// wire    [127:0] axis_slave2_tdata       ;
// wire            axis_slave2_tlast       ;
// wire            axis_slave2_tuser       ;

// wire    [7:0]   cfg_pbus_num            ;
// wire    [4:0]   cfg_pbus_dev_num        ;
// wire    [2:0]   cfg_max_rd_req_size     ;
// wire    [2:0]   cfg_max_payload_size    ;
// wire            cfg_rcb                 ;
//system signal
// wire    [4:0]   smlh_ltssm_state       synthesis PAP_MARK_DEBUG="1";
// wire            core_rst_n             synthesis PAP_MARK_DEBUG="1";
// wire            sync_button_rst_n      synthesis PAP_MARK_DEBUG="1";
// wire            sync_perst_n           synthesis PAP_MARK_DEBUG="1";  
// wire            smlh_link_up           synthesis PAP_MARK_DEBUG="1";
// wire            rdlh_link_up           /* synthesis PAP_MARK_DEBUG="1" */; 
    

// assign axis_slave0_tvalid      = 'd0;
// assign axis_slave0_tlast       = 'd0;
// assign axis_slave0_tuser       = 'd0;
// assign axis_slave0_tdata       = 'd0;
// assign axis_slave1_tvalid      = 'd0;
// assign axis_slave1_tlast       = 'd0;
// assign axis_slave1_tuser       = 'd0;
// assign axis_slave1_tdata       = 'd0;

// pcie_test u_ipsl_pcie_wrap
// (
    // .button_rst_n               (sync_button_rst_n      ),
    // .power_up_rst_n             (sync_perst_n           ),
    // .perst_n                    (sync_perst_n           ),
    //clk and rst
    // .free_clk                   (ref_clk               ),
    // .pclk                       (pclk                   ),      //output
    // .pclk_div2                  (pclk_div2              ),      //output
    // .ref_clk                    (pcie_ref_clk                ),      //output
    // .ref_clk_n                  (ref_clk_n              ),      //input
    // .ref_clk_p                  (ref_clk_p              ),      //input
    // .core_rst_n                 (core_rst_n             ),      //output
    
    //APB interface to  DBI cfg
 //.p_clk                      (ref_clk                ),      //input
    // .p_sel                      (                       ),      //input
    // .p_strb                     (                       ),      //input  [ 3:0]
    // .p_addr                     (                       ),      //input  [15:0]
    // .p_wdata                    (                       ),      //input  [31:0]
    // .p_ce                       (                       ),      //input
    // .p_we                       (                       ),      //input
    // .p_rdy                      (                       ),      //output
    // .p_rdata                    (                       ),      //output [31:0]
    
    //PHY diff signals
    // .rxn                        (rxn                    ),      //input   max[3:0]
    // .rxp                        (rxp                    ),      //input   max[3:0]
    // .txn                        (txn                    ),      //output  max[3:0]
    // .txp                        (txp                    ),      //output  max[3:0]
    
    // .pcs_nearend_loop           (1'b0                   ),      //input
    // .pma_nearend_ploop          (1'b0                   ),      //input
    // .pma_nearend_sloop          (1'b0                   ),      //input
    
   // AXIS master interface
    // .axis_master_tvalid         (axis_master_tvalid     ),      //output
    // .axis_master_tready         (axis_master_tready     ),      //input
    // .axis_master_tdata          (axis_master_tdata      ),      //output [127:0]
    // .axis_master_tkeep          (axis_master_tkeep      ),      //output [3:0]
    // .axis_master_tlast          (axis_master_tlast      ),      //output
    // .axis_master_tuser          (axis_master_tuser      ),      //output [7:0]
    
    //axis slave 0 interface
    // .axis_slave0_tready         (axis_slave0_tready     ),      //output
    // .axis_slave0_tvalid         (axis_slave0_tvalid     ),      //input
    // .axis_slave0_tdata          (axis_slave0_tdata      ),      //input  [127:0]
    // .axis_slave0_tlast          (axis_slave0_tlast      ),      //input
    // .axis_slave0_tuser          (axis_slave0_tuser      ),      //input
    
    //axis slave 1 interface
    // .axis_slave1_tready         (axis_slave1_tready     ),      //output
    // .axis_slave1_tvalid         (axis_slave1_tvalid     ),      //input
    // .axis_slave1_tdata          (axis_slave1_tdata      ),      //input  [127:0]
    // .axis_slave1_tlast          (axis_slave1_tlast      ),      //input
    // .axis_slave1_tuser          (axis_slave1_tuser      ),      //input
    //axis slave 2 interface
    // .axis_slave2_tready         (axis_slave2_tready     ),      //output
    // .axis_slave2_tvalid         (axis_slave2_tvalid     ),      //input
    // .axis_slave2_tdata          (axis_slave2_tdata      ),      //input  [127:0]
    // .axis_slave2_tlast          (axis_slave2_tlast      ),      //input
    // .axis_slave2_tuser          (axis_slave2_tuser      ),      //input
     
    // .pm_xtlh_block_tlp          (                       ),      //output
    
    // .cfg_send_cor_err_mux       (                       ),      //output
    // .cfg_send_nf_err_mux        (                       ),      //output
    // .cfg_send_f_err_mux         (                       ),      //output
    // .cfg_sys_err_rc             (                       ),      //output
    // .cfg_aer_rc_err_mux         (                       ),      //output
    //radm timeout
    // .radm_cpl_timeout           (                       ),      //output
    
    //configuration signals
    // .cfg_max_rd_req_size        (cfg_max_rd_req_size    ),      //output [2:0]
    // .cfg_bus_master_en          (                       ),      //output
    // .cfg_max_payload_size       (cfg_max_payload_size   ),      //output [2:0]
    // .cfg_ext_tag_en             (                       ),      //output
    // .cfg_rcb                    (cfg_rcb                ),      //output
    // .cfg_mem_space_en           (                       ),      //output
    // .cfg_pm_no_soft_rst         (                       ),      //output
    // .cfg_crs_sw_vis_en          (                       ),      //output
    // .cfg_no_snoop_en            (                       ),      //output
    // .cfg_relax_order_en         (                       ),      //output
    // .cfg_tph_req_en             (                       ),      //output [2-1:0]
    // .cfg_pf_tph_st_mode         (                       ),      //output [3-1:0]
    // .rbar_ctrl_update           (                       ),      //output
    // .cfg_atomic_req_en          (                       ),      //output
    
    // .cfg_pbus_num               (cfg_pbus_num           ),      //output [7:0]
    // .cfg_pbus_dev_num           (cfg_pbus_dev_num       ),      //output [4:0]
    
    //debug signals
    // .radm_idle                  (                       ),      //output
    // .radm_q_not_empty           (                       ),      //output
    // .radm_qoverflow             (                       ),      //output
    // .diag_ctrl_bus              (2'b0                   ),      //input   [1:0]
    // .cfg_link_auto_bw_mux       (                       ),      //output              merge cfg_link_auto_bw_msi and cfg_link_auto_bw_int
    // .cfg_bw_mgt_mux             (                       ),      //output              merge cfg_bw_mgt_int and cfg_bw_mgt_msi
    // .cfg_pme_mux                (                       ),      //output              merge cfg_pme_int and cfg_pme_msi
    // .app_ras_des_sd_hold_ltssm  (1'b0                   ),      //input
    // .app_ras_des_tba_ctrl       (2'b0                   ),      //input   [1:0]
    
    // .dyn_debug_info_sel         (4'b0                   ),      //input   [3:0]
    // .debug_info_mux             (                       ),      //output  [132:0]
    
    //system signal
    // .smlh_link_up               (smlh_link_up           ),      //output
    // .rdlh_link_up               (rdlh_link_up           ),      //output
    // .smlh_ltssm_state           (smlh_ltssm_state       )       //output  [4:0]
// );
test_top test_top(
   .pix_clk(pixclk_in),
   .rstn(processing_rst_n),
   //.eth_rst_n_0(eth_rst_n_0)        , //以太网复位信号
   .eth_rgmii_rxc_0(eth_rgmii_rxc_0)    ,
   .eth_rgmii_rx_ctl_0 (eth_rgmii_rx_ctl_0),
	.eth_rgmii_rxd_0(eth_rgmii_rxd_0)    ,  
             
   .eth_rgmii_txc_0 (eth_rgmii_txc_0)   ,
   .eth_rgmii_tx_ctl_0 (eth_rgmii_tx_ctl_0),
   .eth_rgmii_txd_0 (eth_rgmii_txd_0) ,  
	.vs(vs_out),
	.hs(hs_out),
	.de(de_out),
	.i_rgb565({r_out[7:3],g_out[7:2],b_out[7:3]})




   );
endmodule
