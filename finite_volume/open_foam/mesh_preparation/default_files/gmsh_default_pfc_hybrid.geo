
//Fuselage contour
BSpline(2003)={idx_f_int+1:idx_f3+1};Transfinite Curve {2003} = n_h_cent_2 Using Progression p_h_cent_2;
BSpline(4)={idx_f3+1:idx_f4};Transfinite Curve {4} = n_h_inlet_1 Using Progression p_h_inlet_1;
BSpline(5)={idx_f4:idx_f4_1,idx_f5};Transfinite Curve {5} = n_h_inlet_2 Using Progression p_h_inlet_2;
BSpline(6)={idx_f5:idx_f6};Transfinite Curve {6} = n_h_ff_1_fus Using Progression 1/p_h_ff_1_fus;
BSpline(7)={idx_f6:idx_f7};Transfinite Curve {7} = n_h_ff_2_fus Using Progression 1/p_h_ff_2_fus;
BSpline(8)={idx_f7:idx_f9};Transfinite Curve {8} = n_h_rot_fus Using Progression p_h_rot;
BSpline(10)={idx_f9:idx_f11};Transfinite Curve {10} = n_h_gap_fus Using Progression p_h_gap;
BSpline(12)={idx_f11:idx_f13};Transfinite Curve {12} = n_h_stat_fus Using Progression p_h_stat;
BSpline(2014)={idx_f13:idx_f14};Transfinite Curve {2014} = n_h_ff_9_fus Using Progression p_h_ff_9_fus;
BSpline(15)={idx_f14:idx_f15_1-1,idx_f15};Transfinite Curve {15} = n_h_ff_10_fus Using Progression p_h_ff_10_fus;
BSpline(16)={idx_f15:idx_f16};Transfinite Curve {16} = n_h_tail Using Progression p_h_tail;

// Fuselage BL contour
BSpline(2019)={idx_fbl_int:idx_fbl17-1,33001};Transfinite Curve {2019} = n_h_cent_2 Using Progression p_h_cent_2;
BSpline(20)={33001:idx_fbl4-1,34000};Transfinite Curve {20} = n_h_inlet_1 Using Progression p_h_inlet_1;
BSpline(21)={34000:idx_fbl5-1};Transfinite Curve {21} = n_h_inlet_2 Using Progression p_h_inlet_2;
BSpline(300)={idx_fbl5-1,idx_bf5:idx_bf6};Transfinite Curve {300} = n_h_ff_1_fus Using Progression 1/p_h_ff_1_fus;
BSpline(301)={idx_bf6:idx_bf7};Transfinite Curve {301} = n_h_ff_2_fus Using Progression 1/p_h_ff_2_fus;
BSpline(302)={idx_bf7:idx_bf9};Transfinite Curve {302} = n_h_rot_fus Using Progression p_h_rot;
BSpline(304)={idx_bf9:idx_bf11};Transfinite Curve {304} = n_h_gap_fus Using Progression p_h_gap;
BSpline(306)={idx_bf11:idx_bf13};Transfinite Curve {306} = n_h_stat_fus Using Progression p_h_stat;
BSpline(308)={idx_bf13:idx_bf14};Transfinite Curve {308} = n_h_ff_9_fus Using Progression p_h_ff_9_fus;
BSpline(309)={idx_bf14:idx_bf15};Transfinite Curve {309} = n_h_ff_10_fus Using Progression p_h_ff_10_fus;
BSpline(350)={idx_bf15,35001:idx_fbl16-1,101003};Transfinite Curve {350} = n_h_tail Using Progression p_h_tail;

// Fuselage inner line contour
BSpline(2030)={idx_i_int:idx_i17-1,53001};Transfinite Curve {2030} = n_h_fuse_inner_cent Using Progression p_h_fuse_inner_cent;

// domain, horizontal lines
Line(205)={idx_f_int+1,idx_fbl_int};Transfinite Curve {205} = n_v_fbl Using Progression p_v_fbl;
//Line(805)={idx_f2,32000};Transfinite Curve {805} = n_v_fbl Using Progression p_v_fbl;

Line(211)={idx_f3+1,33001};Transfinite Curve {211} = n_v_fbl Using Progression p_v_fbl;
Line(215)={idx_f4,34000};Transfinite Curve {215} = n_v_fbl Using Progression p_v_fbl;
Line(221)={10000,idx_fbl5-1};Transfinite Curve {221} = n_v_fbl Using Progression p_v_fbl;
//Line(402)={idx_fbl3-1,idx_f17};Transfinite Curve {402} = n_v_fbl Using Progression 1/p_v_fbl;
// domain, vertical lines
Line(400)={idx_bnb5,idx_nb5};Transfinite Curve {400} = n_v_nbl Using Progression 1/p_v_nbl;

// nacelle bottom BL line
BSpline(320)={idx_bnb5:idx_bnb6};Transfinite Curve {320} = n_h_ff_1_nac Using Progression 1/p_h_ff_1_nac;
BSpline(321)={idx_bnb6:idx_bnb7};Transfinite Curve {321} = n_h_ff_2_nac Using Progression 1/p_h_ff_2_nac;
BSpline(322)={idx_bnb7:idx_bnb9};Transfinite Curve {322} = n_h_rot_nac Using Progression p_h_rot;
BSpline(324)={idx_bnb9:idx_bnb11};Transfinite Curve {324} = n_h_gap_nac Using Progression p_h_gap;
BSpline(326)={idx_bnb11:idx_bnb13};Transfinite Curve {326} = n_h_stat_nac Using Progression p_h_stat;
BSpline(328)={idx_bnb13:idx_bnb14};Transfinite Curve {328} = n_h_ff_9_nac Using Progression p_h_ff_9_nac;
BSpline(329)={idx_bnb14:idx_bnb15,101002,101001};Transfinite Curve {329} = n_h_ff_10_nac Using Progression p_h_ff_10_nac;

// nacelle bottom line
BSpline(330)={idx_nb5:idx_nb6};Transfinite Curve {330} = n_h_ff_1_nac Using Progression 1/p_h_ff_1_nac;
BSpline(331)={idx_nb6:idx_nb7};Transfinite Curve {331} = n_h_ff_2_nac Using Progression 1/p_h_ff_2_nac;
BSpline(332)={idx_nb7:idx_nb9};Transfinite Curve {332} = n_h_rot_nac Using Progression p_h_rot;
BSpline(334)={idx_nb9:idx_nb11};Transfinite Curve {334} = n_h_gap_nac Using Progression p_h_gap;
BSpline(336)={idx_nb11:idx_nb13};Transfinite Curve {336} = n_h_stat_nac Using Progression p_h_stat;
BSpline(338)={idx_nb13:idx_nb14};Transfinite Curve {338} = n_h_ff_9_nac Using Progression p_h_ff_9_nac;
BSpline(339)={idx_nb14:idx_nb15};Transfinite Curve {339} = n_h_ff_10_nac Using Progression p_h_ff_10_nac;

Line(11003)={idx_nb15,101001};Transfinite Curve {11003} = n_v_nbl Using Progression p_v_nbl;

// nacelle top line
BSpline(600)={idx_nb5,idx_nt50+1:idx_nt51};Transfinite Curve {600} = n_h_nac_1 Using Progression p_h_nac_1;
BSpline(601)={idx_nt51:idx_nt52};Transfinite Curve {601} = n_h_nac_2 Using Progression p_h_nac_2;
BSpline(602)={idx_nt52:idx_nt53};Transfinite Curve {602} = n_h_nac_3 Using Progression p_h_nac_3;
BSpline(603)={idx_nt53:idx_nt54-1,idx_nb15};Transfinite Curve {603} = n_h_nac_4 Using Progression p_h_nac_4;
BSpline(604)={idx_bnb5,idx_bnt50+1:idx_bnt51};Transfinite Curve {604} = n_h_nac_1 Using Progression p_h_nac_1;
BSpline(605)={idx_bnt51:idx_bnt52};Transfinite Curve {605} = n_h_nac_2 Using Progression p_h_nac_2;
BSpline(606)={idx_bnt52:idx_bnt53};Transfinite Curve {606} = n_h_nac_3 Using Progression p_h_nac_3;
BSpline(607)={idx_bnt53:idx_bnt54,101000,101001};Transfinite Curve {607} = n_h_nac_4 Using Progression p_h_nac_4;

// domain above top nacelle
Line(700)={idx_nt51,idx_bnt51};Transfinite Curve {700} = n_v_nbl Using Progression p_v_nbl;
Line(701)={idx_nt52,idx_bnt52};Transfinite Curve {701} = n_v_nbl Using Progression p_v_nbl;
Line(702)={idx_nt53,idx_bnt53};Transfinite Curve {702} = n_v_nbl Using Progression p_v_nbl;
//Line(703)={idx_nb15,idx_bnt54};Transfinite Curve {703} = n_v_nbl Using Progression p_v_nbl;

// FF stage vertical lines
BSpline(500)={idx_f6,20001:idx_67_1-1,idx_bf6};Transfinite Curve {500} = n_v_fbl Using Progression p_v_fbl;
BSpline(503)={idx_bnb6,20301:idx_67_4-1,idx_nb6};Transfinite Curve {503} = n_v_nbl Using Progression 1/p_v_nbl;
BSpline(504)={idx_f7,21001:idx_78_1-1,idx_bf7};Transfinite Curve {504} = n_v_fbl Using Progression p_v_fbl;
BSpline(505)={idx_bf7,21101:idx_78_2-1,idx_m7,21201:idx_78_3-1,idx_bnb7};Transfinite Curve {505} = n_v_duct_rot_in Using Progression p_v_duct_rot_in;
BSpline(507)={idx_bnb7,21301:idx_78_4-1,idx_nb7};Transfinite Curve {507} = n_v_nbl Using Progression 1/p_v_nbl;

BSpline(512)={idx_f9,23001:idx_910_1-1,idx_bf9};Transfinite Curve {512} = n_v_fbl Using Progression p_v_fbl;
BSpline(513)={idx_bf9,23101:idx_910_2-1,idx_m9,23201:idx_910_3-1,idx_bnb9};Transfinite Curve {513} = n_v_duct_rot_out Using Progression p_v_duct_rot_out;
BSpline(515)={idx_bnb9,23301:idx_910_4-1,idx_nb9};Transfinite Curve {515} = n_v_nbl Using Progression 1/p_v_nbl;

BSpline(520)={idx_f11,25001:idx_1112_1-1,idx_bf11};Transfinite Curve {520} = n_v_fbl Using Progression p_v_fbl;
BSpline(521)={idx_bf11,25101:idx_1112_2-1,idx_m11,25201:idx_1112_3-1,idx_bnb11};Transfinite Curve {521} = n_v_duct_stat_in Using Progression p_v_duct_stat_in;
BSpline(523)={idx_bnb11,25301:idx_1112_4-1,idx_nb11};Transfinite Curve {523} = n_v_nbl Using Progression 1/p_v_nbl;

BSpline(528)={idx_f13,27001:idx_1314_1-1,idx_bf13};Transfinite Curve {528} = n_v_fbl Using Progression p_v_fbl;
BSpline(529)={idx_bf13,27101:idx_1314_2-1,idx_m13,27201:idx_1314_3-1,idx_bnb13};Transfinite Curve {529} = n_v_duct_stat_out Using Progression p_v_duct_stat_out;
BSpline(531)={idx_bnb13,27301:idx_1314_4-1,idx_nb13};Transfinite Curve {531} = n_v_nbl Using Progression 1/p_v_nbl;
BSpline(535)={idx_bnb14,28301:idx_1415_4-1,idx_nb14};Transfinite Curve {535} = n_v_nbl Using Progression 1/p_v_nbl;

BSpline(532)={idx_f14,28001:idx_1415_1-1,idx_bf14};Transfinite Curve {532} = n_v_fbl Using Progression p_v_fbl;
Line(540)={idx_f15,idx_bf15};Transfinite Curve {540} = n_v_fbl Using Progression p_v_fbl;

// domain inlet
Line(1302)={62000,66012};Transfinite Curve {1302} = n_cent_up Using Progression p_cent_up;

Line(1308)={idx_f16,101003};Transfinite Curve {1308} = n_v_fbl Using Progression p_v_fbl;
Line(1309)={101003,66002};Transfinite Curve {1309} = n_h_rear_1 Using Progression p_h_rear_1;
Line(1310)={66002,66001};Transfinite Curve {1310} = n_h_rear_2 Using Progression p_h_rear_2;
Line(1320)={66012,66013};Transfinite Curve {1320} = n_rear_domain Using Progression p_rear_domain;
Line(1327)={53001,66010};Transfinite Curve {1327} = n_h_fuse_inner_back Using Progression p_h_fuse_inner_back;
Line(1322)={66010,66011};Transfinite Curve {1322} = n_h_rear_2 Using Progression p_h_rear_2;
Line(1324)={66002,66010};Transfinite Curve {1324} = n_v_low Using Progression p_v_low;
Line(1328)={66010,66012};Transfinite Curve {1328} = n_v_up_mid Using Progression p_v_up_mid;
Line(1333)={66001,66003};Transfinite Curve {1333} = 2;
Line(1334)={66003,66011};Transfinite Curve {1334} = n_rear_low Using Progression p_rear_low;
Line(1338)={66011,66013};Transfinite Curve {1338} = n_rear_up Using Progression p_rear_up;

Line(1303)={idx_i_int,62000};Transfinite Curve {1303} = n_v_int_up Using Progression p_v_int_up;
Line(1304)={idx_fbl_int,idx_i_int};Transfinite Curve {1304} = n_v_int_mid Using Progression p_v_int_mid;

Curve Loop(1) = {1327, 1328, -1302, -1303, 2030};
Plane Surface(1) = {1};
Curve Loop(100) = {1324, -1327, -2030, -1304, 2019, 20, 21, 300, 301, 505, -321, -320, 604, 605, 606, 607, -329, -328, -529, 308, 309, 350, 1309};
Plane Surface(100) = {100};
Curve Loop(2) = {1328, 1320, -1338, -1322};
Plane Surface(2) = {2};//+
Curve Loop(3) = {1324, 1322, -1334, -1333, -1310};
Plane Surface(3) = {3};
Curve Loop(40) = {21, -221, -5, 215};
Plane Surface(40) = {40};
Curve Loop(41) = {20, -215, -4, 211};
Plane Surface(41) = {41};
Curve Loop(42) = {211, -2019, -205, 2003};
Plane Surface(42) = {42};
Curve Loop(46) = {604, -700, -600, -400};
Plane Surface(46) = {46};
Curve Loop(47) = {601, 701, -605, -700};
Plane Surface(47) = {47};
Curve Loop(48) = {602, 702, -606, -701};
Plane Surface(48) = {48};
Curve Loop(49) = {603, 11003, -607, -702};
Plane Surface(49) = {49};
Curve Loop(50) = {15, 540, -309, -532};
Plane Surface(50) = {50};
Curve Loop(55) = {338, -535, -328, 531};
Plane Surface(55) = {55};
Curve Loop(82) = {321, 507, -331, -503};
Plane Surface(82) = {82};
Curve Loop(86) = {330, -503, -320, 400};
Plane Surface(86) = {86};
Curve Loop(87) = {329, -11003, -339, -535};
Plane Surface(87) = {87};
Curve Loop(88) = {2014, 532, -308, -528};
Plane Surface(88) = {88};
Curve Loop(79) = {301, -504, -7, 500};
Plane Surface(79) = {79};
Curve Loop(83) = {6, 500, -300, -221};
Plane Surface(83) = {83};
Curve Loop(90) = {350, -1308, -16, 540};
Plane Surface(90) = {90};//+
Curve Loop(91) = {322, 515, -332, -507};
Plane Surface(91) = {91};
Curve Loop(92) = {324, 523, -334, -515};
Plane Surface(92) = {92};
Curve Loop(93) = {326, 531, -336, -523};
Plane Surface(93) = {93};
Curve Loop(94) = {528, -306, -520, 12};
Plane Surface(94) = {94};
Curve Loop(95) = {304, -520, -10, 512};
Plane Surface(95) = {95};
Curve Loop(96) = {8, 512, -302, -504};
Plane Surface(96) = {96};
Curve Loop(97) = {505, 322, -513, -302};
Plane Surface(97) = {97};
Curve Loop(98) = {304, 521, -324, -513};
Plane Surface(98) = {98};
Curve Loop(99) = {306, 529, -326, -521};
Plane Surface(99) = {99};

Transfinite Surface{40:42,46:50,55,79,82,83,86:88,90:96};
Recombine Surface{40:50,55,79,82,83,86:88,90:96};

Extrude {0, 0, 10} {
  Surface{1:100}; Layers{1}; Recombine;
}
Physical Surface("wedge_left") = {1:3,40:42,46:50,55,79,82,83,86:88,90:100};
Physical Surface("axi") = {11745,11078,11422};
Physical Surface("inlet") = {11025,11669,11140};
Physical Surface("atmosphere") = {11021,11043};
Physical Surface("outlet") = {11047,11070,11074};
Physical Surface("fuse_center") = {11144};
Physical Surface("fuse_tail") = {11426};
Physical Surface("fuse_sweep") = {11118,11096};
Physical Surface("fuse_hub_inlet") = {11294,11330};
Physical Surface("fuse_hub_nozzle") = {11396,11242};
Physical Surface("fuse_hub_rotor") = {11550};
Physical Surface("fuse_hub_gap") = {11536};
Physical Surface("fuse_hub_stator") = {11518};
Physical Surface("nac_cowling") = {11220,11198,11176,11162};
Physical Surface("nac_nozzle") = {11264,11382};
Physical Surface("nac_inlet") = {11352,11316};
Physical Surface("nac_rotor") = {11448};
Physical Surface("nac_gap") = {11470};
Physical Surface("nac_stator") = {11492};
Physical Volume("rotor") = {20,25,26};
Physical Volume("stator") = {22,23,28};
Physical Volume("rest") = {1:19,21,24,27,29};
Physical Surface("wedge_right") = {11052, 11030, 11079, 11746, 11431, 11255, 11409, 11519, 11541, 11563, 11299, 11585, 11607, 11629, 11277, 11387, 11233, 11211, 11497, 11475, 11453, 11321, 11365, 11343, 11189, 11167, 11101, 11123, 11145};

Geometry.Tolerance = 1e-20;
Mesh.CharacteristicLengthFactor = 1;
Mesh.CharacteristicLengthMin = 1e-8;
Mesh.Optimize = 1;