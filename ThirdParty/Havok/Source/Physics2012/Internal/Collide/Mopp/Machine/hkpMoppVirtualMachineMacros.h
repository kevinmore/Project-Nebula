/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Common/Base/Math/Vector/hkIntVector.h>

//
//	Macros for virtual machines:
//  Assumptions:<br>
//  The machine defines the following inline function<br>
//	   - void hkMoppXXXXXXXXxxVirtualMachine::addHit(unsigned int key, const unsigned int properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES])
//
//	The following variables are used:<br>
//		- PC:		the current program counter
//		- offsetl   an integer variable
//		- offseth   an integer variable
//		- query     the const query input holding an integer m_primitiveOffset and m_properties[1] member
//      - scaledQuery a temporary copy of query to hold a modified version of the query

//For PPu on PlayStation(R)3 and Xbox 360 using a lookup table eliminated LHS when converting form int to float
#if defined(HK_PLATFORM_XBOX360) || ( defined(HK_PLATFORM_PS3_PPU) && !defined(HK_PLATFORM_SPU) )
#define HK_UINT8_TO_REAL( x ) hkReal( hkUInt8ToFloat32[ (x) ] )
#else
#define HK_UINT8_TO_REAL( x ) hkReal ( (x) )
#endif

#if defined(HK_PLATFORM_SPU)
#define PC0 commandVec.u[0]
#define PC1 PC1_var
#define PC2 PC2_var
#define PC3 PC3_var
#define PC4 commandVec.u[4]
#define PC5 commandVec.u[5]
#define PC6 commandVec.u[6]
#define PC7 commandVec.u[7]
#define PC8 commandVec.u[8]
#define PC9 commandVec.u[9]
#define PC10 commandVec.u[10]
#define PC11 commandVec.u[11]
#define PC12 commandVec.u[12]
#define PC13 commandVec.u[13]
#define PC14 commandVec.u[14]
#define PC15 commandVec.u[15]

#define PC1_float PC1_float_var
#define PC2_float PC2_float_var
// PC3 is rarely used so we do not preload it
#define PC3_float HK_UINT8_TO_REAL(PC3)


#define HK_MOPP_LOAD_PC_INTEGERONLY() \
	hkQuadUcharUnion commandVec; \
	/* PC is not aligned */ \
	hkIntVector tempIV; tempIV.loadNotAligned<4>(reinterpret_cast<const hkUint32*>(PC)); \
	commandVec.q = tempIV.m_quad; \
	const hkpMoppCommands command = hkpMoppCommands(PC0); \
	hkUint8 PC1_var = commandVec.u[1]; hkUint8 PC2_var = commandVec.u[2]; hkUint8 PC3_var = commandVec.u[3]


#define HK_MOPP_LOAD_PC() \
	HK_MOPP_LOAD_PC_INTEGERONLY(); \
	const float PC1_float_var = HK_UINT8_TO_REAL(PC1); \
	const float PC2_float_var = HK_UINT8_TO_REAL(PC2)

// Currently, the _float variables do not require reloading
#define HK_MOPP_RELOAD_PC() \
	tempIV.loadNotAligned<4>(reinterpret_cast<const hkUint32*>(PC)); \
	commandVec.q = tempIV.m_quad; \
	PC1_var = commandVec.u[1]; PC2_var = commandVec.u[2]; PC3_var = commandVec.u[3]

#else

#define PC0 PC[0]
#define PC1 PC1_var
#define PC2 PC2_var
#define PC3 PC3_var
#define PC4 PC[4]
#define PC5 PC[5]
#define PC6 PC[6]
#define PC7 PC[7]
#define PC8 PC[8]
#define PC9 PC[9]
#define PC10 PC[10]
#define PC11 PC[11]
#define PC12 PC[12]
#define PC13 PC[13]
#define PC14 PC[14]
#define PC15 PC[15]

#define PC1_float HK_UINT8_TO_REAL(PC1)
#define PC2_float HK_UINT8_TO_REAL(PC2)
// PC3 is rarely used so we do not preload it
#define PC3_float HK_UINT8_TO_REAL(PC3)

#define HK_MOPP_LOAD_PC_INTEGERONLY() \
	const hkpMoppCommands command = hkpMoppCommands(PC0); \
	hkUint8 PC1_var = PC[1]; hkUint8 PC2_var = PC[2]; hkUint8 PC3_var = PC[3]

#define HK_MOPP_LOAD_PC() HK_MOPP_LOAD_PC_INTEGERONLY()

// Currently, the _float variables do not require reloading
#define HK_MOPP_RELOAD_PC() \
	PC1_var = PC[1]; PC2_var = PC[2]; PC3_var = PC[3]

#endif

#define	HK_MOPP_JUMP_MACRO								\
		case HK_MOPP_JUMP8:								\
			{											\
				offsetl = PC1;							\
				PC += 2;								\
				PC += offsetl;							\
				continue;								\
			}											\
		case HK_MOPP_JUMP16:							\
			{											\
				offseth = PC1;							\
				offsetl = PC2;							\
				PC += 3;								\
				PC += (offseth << 8) + offsetl;			\
				continue;								\
			}											\
		case HK_MOPP_JUMP24:							\
			{											\
				offseth = PC1;							\
				const unsigned int offsetm = PC2;		\
				offsetl = PC3;							\
				PC += 4;								\
				PC += (offseth << 16) + (offsetm << 8) + offsetl;	\
				continue;								\
			}											\
		case HK_MOPP_JUMP32:							\
			{											\
				offseth = PC1;							\
				const unsigned int offsetmh = PC2;		\
				const unsigned int offsetml = PC3;		\
				offsetl = PC4;							\
				PC += 5;								\
				PC += (offseth << 24) + (offsetmh << 16) + (offsetml << 8) + offsetl;	\
				continue;								\
			}



#define HK_MOPP_REOFFSET_MACRO									\
		case HK_MOPP_TERM_REOFFSET8:								\
			{													\
				const unsigned int offset = PC1;				\
				if ( query != &scaledQuery)						\
				{												\
					scaledQuery = *query;						\
					query = &scaledQuery;						\
				}												\
				scaledQuery.m_primitiveOffset = scaledQuery.m_primitiveOffset + offset;		\
				PC+=2;											\
				continue;										\
			}													\
		case HK_MOPP_TERM_REOFFSET16:							\
			{													\
				const unsigned int offset = (PC1<<8) + PC2;	\
				if ( query != &scaledQuery)						\
				{												\
					scaledQuery = *query;						\
					query = &scaledQuery;						\
				}												\
				scaledQuery.m_primitiveOffset = scaledQuery.m_primitiveOffset + offset;		\
				PC+=3;											\
				continue;										\
			}													\
		case HK_MOPP_TERM_REOFFSET32:							\
			{													\
				const unsigned int offset = (PC1<<24) + (PC2<<16) + (PC3<<8) + PC4;						\
				if ( query != &scaledQuery)						\
				{												\
					scaledQuery = *query;						\
					query = &scaledQuery;						\
				}												\
				scaledQuery.m_primitiveOffset = offset;			\
				PC+=5;											\
				continue;										\
			}

#define		HK_MOPP_TERMINAL_COMMON_MACRO			\
		case HK_MOPP_RETURN:				\
			{								\
			goto end_of_function;		\
			}								\
		case HK_MOPP_TERM8:					\
			{								\
			offsetl = PC1;			\
			goto add_Terminal;			\
			}								\
		case HK_MOPP_TERM16:				\
			{								\
			offseth = PC1;			\
			offseth <<=8;				\
			offsetl = PC2;			\
			offsetl += offseth;			\
			goto add_Terminal;			\
			}								\
		case HK_MOPP_TERM24:				\
			{								\
			offsetl = PC1;			\
			offsetl <<=16;				\
			offseth = PC2;			\
			offseth <<=8;				\
			offseth += PC3;			\
			offsetl += offseth;			\
			goto add_Terminal;			\
			}								\
		case HK_MOPP_TERM32:				\
			{								\
			offsetl = PC1;			\
			offsetl <<=24;				\
			offseth = PC2;			\
			offseth <<=16;				\
			offsetl += offseth;			\
			offseth = PC3;			\
			offsetl += PC4;			\
			offseth <<=8;				\
			offsetl += offseth;			\
			goto add_Terminal;			\
			}								\
		case HK_MOPP_TERM4_0:				\
		case HK_MOPP_TERM4_1:				\
		case HK_MOPP_TERM4_2:				\
		case HK_MOPP_TERM4_3:				\
		case HK_MOPP_TERM4_4:				\
		case HK_MOPP_TERM4_5:				\
		case HK_MOPP_TERM4_6:				\
		case HK_MOPP_TERM4_7:				\
		case HK_MOPP_TERM4_8:				\
		case HK_MOPP_TERM4_9:				\
		case HK_MOPP_TERM4_A:				\
		case HK_MOPP_TERM4_B:				\
		case HK_MOPP_TERM4_C:				\
		case HK_MOPP_TERM4_D:				\
		case HK_MOPP_TERM4_E:				\
		case HK_MOPP_TERM4_F:				\
		case HK_MOPP_TERM4_10:				\
		case HK_MOPP_TERM4_11:				\
		case HK_MOPP_TERM4_12:				\
		case HK_MOPP_TERM4_13:				\
		case HK_MOPP_TERM4_14:				\
		case HK_MOPP_TERM4_15:				\
		case HK_MOPP_TERM4_16:				\
		case HK_MOPP_TERM4_17:				\
		case HK_MOPP_TERM4_18:				\
		case HK_MOPP_TERM4_19:				\
		case HK_MOPP_TERM4_1A:				\
		case HK_MOPP_TERM4_1B:				\
		case HK_MOPP_TERM4_1C:				\
		case HK_MOPP_TERM4_1D:				\
		case HK_MOPP_TERM4_1E:				\
		case HK_MOPP_TERM4_1F:				\
		offsetl = command - HK_MOPP_TERM4_0;\

#define		HK_MOPP_TERMINAL_MACRO						\
			HK_MOPP_TERMINAL_COMMON_MACRO				\
			{											\
add_Terminal:											\
				offsetl += query->m_primitiveOffset;	\
				addHit(offsetl,query->m_properties);	\
				goto end_of_function;					\
			}											\

#define		HK_MOPP_CHUNK_TERMINAL_MACRO							\
			HK_MOPP_TERMINAL_COMMON_MACRO							\
			{														\
			add_Terminal:											\
				offsetl += query->m_primitiveOffset;				\
				unsigned int chunkId = chunkOffset >> HK_MOPP_LOG_CHUNK_SIZE; \
				unsigned int keyRemap = ( (chunkId << 8) & m_reindexingMask ) | offsetl;\
				addHit(keyRemap, query->m_properties );				\
				goto end_of_function;								\
			}														\

#define		HK_MOPP_PROPERTY_MACRO				\
		case HK_MOPP_PROPERTY8_0:				\
		case HK_MOPP_PROPERTY8_1:				\
		case HK_MOPP_PROPERTY8_2:				\
		case HK_MOPP_PROPERTY8_3:				\
			{									\
				unsigned int property; property = PC0 - HK_MOPP_PROPERTY8_0;	\
				unsigned int value; value = PC1;						\
				PC += 2;												\
				scaledQuery.m_properties[property] = value;				\
propertyCopy:															\
				const unsigned int v = scaledQuery.m_properties[0];					\
				if ( query != &scaledQuery)								\
				{														\
					scaledQuery = *query;								\
					query = &scaledQuery;								\
				}														\
				scaledQuery.m_properties[0] = v;						\
				continue;												\
			}															\
		case HK_MOPP_PROPERTY16_0:				\
		case HK_MOPP_PROPERTY16_1:				\
		case HK_MOPP_PROPERTY16_2:				\
		case HK_MOPP_PROPERTY16_3:				\
			{									\
				const unsigned int property = PC0 - HK_MOPP_PROPERTY16_0;	\
				const unsigned int value = (PC1<<8) + PC2;			\
				scaledQuery.m_properties[property] = value;				\
				PC += 3;												\
				goto propertyCopy;										\
			}															\
		case HK_MOPP_PROPERTY32_0:				\
		case HK_MOPP_PROPERTY32_1:				\
		case HK_MOPP_PROPERTY32_2:				\
		case HK_MOPP_PROPERTY32_3:				\
			{									\
				const unsigned int property = PC0 - HK_MOPP_PROPERTY32_0;	\
				const unsigned int value = (PC1<<24) + (PC2<<16) + (PC3<<8) + PC4;						\
				scaledQuery.m_properties[property] = value;				\
				PC += 5;												\
				goto propertyCopy;										\
			}														
							

#define	HK_MOPP_DEFAULT_MACRO								\
		default:											\
			/* Don't use HK_ERROR, since it adds a 512 buffer to the program stack. This is bad since this can appear in recursive functions.*/ \
			/* HK_ERROR(0x1298fedd, "Unknown command - This mopp data has been corrupted (check for memory trashing), or an hkpMoppBvTreeShape has been pointed at invalid mopp data.\n"); */ \
			HK_BREAKPOINT(0x1298fedd);

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
