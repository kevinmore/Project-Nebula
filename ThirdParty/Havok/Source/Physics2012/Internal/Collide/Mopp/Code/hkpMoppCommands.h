/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_COMMANDS_H
#define HK_COLLIDE2_MOPP_COMMANDS_H

#define HK_MOPP_RESOLUTION 255

// MOPP Code commands and arguments definition
enum HK_MOPP_SPLIT_DIRECTIONS {
	HK_MOPP_SD_X,
	HK_MOPP_SD_Y,
	HK_MOPP_SD_Z,
	HK_MOPP_SD_YZ,
	HK_MOPP_SD_YMZ,
	HK_MOPP_SD_XZ,
	HK_MOPP_SD_XMZ,
	HK_MOPP_SD_XY,
	HK_MOPP_SD_XMY,
	HK_MOPP_SD_XYZ,
	HK_MOPP_SD_XYMZ,
	HK_MOPP_SD_XMYZ,
	HK_MOPP_SD_XMYMZ,
	HK_MOPP_SD_MAX
};

enum
{
	HK_MOPP_SCALE0 = 0x00,
	HK_MOPP_MAX_TERM4 = 0x20
};

enum hkpMoppCommands
{
	// 0x00 - 0x04
	HK_MOPP_RETURN   = 0x00,					// return from the MOPP immediately
	// HK_MOPP_SCALE0 = 0x00					// followed by 3(xyz)*8bit new offset and 1 byte how many bits shifted
	HK_MOPP_SCALE1	= 0x01 + HK_MOPP_SCALE0,	// Scale command with explicit right shift of 1	(divide scale by 2)
	HK_MOPP_SCALE2	= 0x02 + HK_MOPP_SCALE0,	// Scale command with explicit right shift of 2 (divide scale by 4)
	HK_MOPP_SCALE3	= 0x03 + HK_MOPP_SCALE0,	// Scale command with explicit right shift of 3 (divide scale by 8)
	HK_MOPP_SCALE4	= 0x04 + HK_MOPP_SCALE0,	// Scale command with explicit right shift of 4 (divide scale by 16)
											// TODO - implement a small jump (4 bit) - may be better to move JUMPs to 0x80 (TODO check if needed)
	// 0x05 - 0x08
	HK_MOPP_JUMP8	= 0x01 + HK_MOPP_SCALE4,		// followed by 8bit rel offset
	HK_MOPP_JUMP16	= 0x01 + HK_MOPP_JUMP8,	// followed by 8bit high and 8 bit low
	HK_MOPP_JUMP24	= 0x02 + HK_MOPP_JUMP8,	// followed by 8bit high, 8bit med and 8 bit low
	HK_MOPP_JUMP32	= 0x03 + HK_MOPP_JUMP8,	// dito

	// 0x09 - 0x0c
	HK_MOPP_TERM_REOFFSET8  = 0x01 + HK_MOPP_JUMP32,
	HK_MOPP_TERM_REOFFSET16 = 0x01 + HK_MOPP_TERM_REOFFSET8,
	HK_MOPP_TERM_REOFFSET32 = 0x02 + HK_MOPP_TERM_REOFFSET8,

	HK_MOPP_JUMP_CHUNK = 0x04 + HK_MOPP_JUMP32,				// deprecated, see HK_MOPP_JUMP_CHUNK32
	HK_MOPP_DATA_OFFSET = 0x01 + HK_MOPP_JUMP_CHUNK,

	// 0x10 - 0x1d
	HK_MOPP_SPLIT_X     = HK_MOPP_SD_X     + 0x10,			// followed by 2*8bit planeDistance and 8bit offset to right branch
	HK_MOPP_SPLIT_Y     = HK_MOPP_SD_Y     + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_Z     = HK_MOPP_SD_Z     + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_YZ    = HK_MOPP_SD_YZ    + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_YMZ   = HK_MOPP_SD_YMZ   + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_XZ    = HK_MOPP_SD_XZ    + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_XMZ   = HK_MOPP_SD_XMZ   + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_XY    = HK_MOPP_SD_XY    + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_XMY   = HK_MOPP_SD_XMY   + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_XYZ   = HK_MOPP_SD_XYZ   + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_XYMZ  = HK_MOPP_SD_XYMZ  + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_XMYZ  = HK_MOPP_SD_XMYZ  + HK_MOPP_SPLIT_X,
	HK_MOPP_SPLIT_XMYMZ = HK_MOPP_SD_XMYMZ + HK_MOPP_SPLIT_X,

	// 0x20 - 0x2d
	HK_MOPP_SINGLE_SPLIT_X     = HK_MOPP_SD_X		+ 0x20,			// followed by 2*8bit planeDistance and 8bit offset to right branch
	HK_MOPP_SINGLE_SPLIT_Y     = HK_MOPP_SD_Y		+ HK_MOPP_SINGLE_SPLIT_X,
	HK_MOPP_SINGLE_SPLIT_Z     = HK_MOPP_SD_Z		+ HK_MOPP_SINGLE_SPLIT_X,


	HK_MOPP_SPLIT_JUMP_X	= 0x01			+ HK_MOPP_SINGLE_SPLIT_Z,			// followed by 2*8bit planeDistance and 8bit offset to right branch
	HK_MOPP_SPLIT_JUMP_Y  = HK_MOPP_SD_Y	+ HK_MOPP_SPLIT_JUMP_X,
	HK_MOPP_SPLIT_JUMP_Z  = HK_MOPP_SD_Z	+ HK_MOPP_SPLIT_JUMP_X,


	HK_MOPP_DOUBLE_CUT_X     = 0x01		      + HK_MOPP_SPLIT_JUMP_Z,				// followed by 8 bit planeDistance
	HK_MOPP_DOUBLE_CUT_Y     = HK_MOPP_SD_Y     + HK_MOPP_DOUBLE_CUT_X,
	HK_MOPP_DOUBLE_CUT_Z     = HK_MOPP_SD_Z     + HK_MOPP_DOUBLE_CUT_X,

	HK_MOPP_DOUBLE_CUT24_X     = 0x01		    + HK_MOPP_DOUBLE_CUT_Z,				// followed by 24 bit planeDistances
	HK_MOPP_DOUBLE_CUT24_Y     = HK_MOPP_SD_Y     + HK_MOPP_DOUBLE_CUT24_X,
	HK_MOPP_DOUBLE_CUT24_Z     = HK_MOPP_SD_Z     + HK_MOPP_DOUBLE_CUT24_X,

	
	HK_MOPP_TERM4_0 = 0x30,
	HK_MOPP_TERM4_1 = 0x01 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_2 = 0x02 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_3 = 0x03 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_4 = 0x04 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_5 = 0x05 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_6 = 0x06 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_7 = 0x07 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_8 = 0x08 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_9 = 0x09 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_A = 0x0a + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_B = 0x0b + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_C = 0x0c + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_D = 0x0d + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_E = 0x0e + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_F = 0x0f + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_10 = 0x10 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_11 = 0x11 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_12 = 0x12 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_13 = 0x13 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_14 = 0x14 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_15 = 0x15 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_16 = 0x16 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_17 = 0x17 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_18 = 0x18 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_19 = 0x19 + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_1A = 0x1a + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_1B = 0x1b + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_1C = 0x1c + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_1D = 0x1d + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_1E = 0x1e + HK_MOPP_TERM4_0,
	HK_MOPP_TERM4_1F = 0x1f + HK_MOPP_TERM4_0,

	// 0x50 - 0x58
	HK_MOPP_TERM8   = 0x01 + HK_MOPP_TERM4_1F,
	HK_MOPP_TERM16  = 0x01 + HK_MOPP_TERM8,
	HK_MOPP_TERM24  = 0x02 + HK_MOPP_TERM8,
	HK_MOPP_TERM32  = 0x03 + HK_MOPP_TERM8,

	HK_MOPP_NTERM8  = 0x04 + HK_MOPP_TERM8, // followed by  8bit numberOfTerminals n, followed by n *  8bit terminal ids
	HK_MOPP_NTERM16 = 0x05 + HK_MOPP_TERM8, // followed by 16bit numberOfTerminals n, followed by n * 16bit
	HK_MOPP_NTERM24 = 0x06 + HK_MOPP_TERM8, // followed by 24bit numberOfTerminals n, followed by n * 24bit
	HK_MOPP_NTERM32 = 0x07 + HK_MOPP_TERM8, // followed by 32bit numberOfTerminals n, followed by n * 32bit

	// 0x60 - 0x70
	HK_MOPP_PROPERTY8_0	= 0x10 + HK_MOPP_TERM8,
	HK_MOPP_PROPERTY8_1	= 0x01 + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY8_2	= 0x02 + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY8_3	= 0x03 + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY16_0	= 0x04 + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY16_1	= 0x05 + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY16_2	= 0x06 + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY16_3	= 0x07 + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY32_0	= 0x08 + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY32_1	= 0x09 + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY32_2	= 0x0A + HK_MOPP_PROPERTY8_0,
	HK_MOPP_PROPERTY32_3	= 0x0B + HK_MOPP_PROPERTY8_0,

	// 0x70 - 0x80
	HK_MOPP_JUMP_CHUNK32	= 0x10 + HK_MOPP_PROPERTY8_0,

};


#endif // HK_COLLIDE2_MOPP_COMMANDS_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
