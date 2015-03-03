/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/KeyCode.h>

// Registration function is at the end of the file

static void hkMotionState_0_to_1(hkDataObject& obj)
{
	// HKV-1215
	obj["timeFactor"] = 1.0f;
}

namespace hkxVertexDescriptionElementDecl_1_to_2_Util
{
	enum OldDataType
	{
		OLD_HKX_DT_NONE = 0,
		OLD_HKX_DT_UINT8, 
		OLD_HKX_DT_INT16, 
		OLD_HKX_DT_UINT32,
		OLD_HKX_DT_FLOAT,
		OLD_HKX_DT_FLOAT2, 
		OLD_HKX_DT_FLOAT3, 
		OLD_HKX_DT_FLOAT4  
	};

	enum DataType
	{
		HKX_DT_NONE = 0,
		HKX_DT_UINT8, 
		HKX_DT_INT16, 
		HKX_DT_UINT32,
		HKX_DT_FLOAT
	};
}
static void hkxVertexDescriptionElementDecl_1_to_2(hkDataObject& obj)
{
	hkUint16 dataType = hkUint16(obj["type"].asInt());

	switch( dataType )
	{
	case hkxVertexDescriptionElementDecl_1_to_2_Util::OLD_HKX_DT_FLOAT:
		{
			obj["type"] = hkxVertexDescriptionElementDecl_1_to_2_Util::HKX_DT_FLOAT;
			obj["numElements"] = 1; 
			break;
		}
	case hkxVertexDescriptionElementDecl_1_to_2_Util::OLD_HKX_DT_FLOAT2:
		{
			obj["type"] = hkxVertexDescriptionElementDecl_1_to_2_Util::HKX_DT_FLOAT;
			obj["numElements"] = 2; 
			break;
		}
	case hkxVertexDescriptionElementDecl_1_to_2_Util::OLD_HKX_DT_FLOAT3:
		{
			obj["type"] = hkxVertexDescriptionElementDecl_1_to_2_Util::HKX_DT_FLOAT;
			obj["numElements"] = 3; 
			break;
		}
	case hkxVertexDescriptionElementDecl_1_to_2_Util::OLD_HKX_DT_FLOAT4:
		{
			obj["type"] = hkxVertexDescriptionElementDecl_1_to_2_Util::HKX_DT_FLOAT;
			obj["numElements"] = 4; 
			break;
		}
	case hkxVertexDescriptionElementDecl_1_to_2_Util::OLD_HKX_DT_INT16:
		{
			obj["type"] = hkxVertexDescriptionElementDecl_1_to_2_Util::HKX_DT_INT16;
			obj["numElements"] = 2; 
			break;
		}
	case hkxVertexDescriptionElementDecl_1_to_2_Util::OLD_HKX_DT_NONE:
		{
			obj["type"] = hkxVertexDescriptionElementDecl_1_to_2_Util::HKX_DT_NONE;
			obj["numElements"] = 0; 
			break;
		}
	case hkxVertexDescriptionElementDecl_1_to_2_Util::OLD_HKX_DT_UINT32:
		{
			obj["type"] = hkxVertexDescriptionElementDecl_1_to_2_Util::HKX_DT_UINT32;
			obj["numElements"] = 1; 
			break;
		}
	case hkxVertexDescriptionElementDecl_1_to_2_Util::OLD_HKX_DT_UINT8:
		{
			obj["type"] = hkxVertexDescriptionElementDecl_1_to_2_Util::HKX_DT_UINT8;
			obj["numElements"] = 4; 
			break;
		}
	default:
		{
			HK_ASSERT(0x5bca7459, 0);
			break;
		}
	}
}

void HK_CALL registerCommonPatches_2010_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_1/hkPatches_2010_1.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
