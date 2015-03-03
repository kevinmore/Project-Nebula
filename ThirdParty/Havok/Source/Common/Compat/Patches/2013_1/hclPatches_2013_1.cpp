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

static bool udpateTriangleFlips(hkDataArray& triangleFlipsSrc, hkDataArray& triangleFlipsDst)
{
	// Note: Presuming that in the case that the cloth-data is loaded on a big-endian platform
	// that is was initially generated on a little-endian platform (since this is the majority 
	// case i.e., win32, x64) and therefore needs to be fixed to take into account endianness. 
	
	// Note: there might be extra unused bytes
	int numInt32 = triangleFlipsSrc.getSize();
	triangleFlipsDst.setSize(numInt32*4);

	bool reexecuteData = false;
	for (int i = 0; i<numInt32; i++)
	{
		hkUint32 triangleFlipAsInt = hkUint32( triangleFlipsSrc[i].asInt() );
		for(int b=0; b<4; b++)
		{
			hkUint8 byteValue = hkUint8( (triangleFlipAsInt >> (8*b) ) & 0xFF );
#if (HK_ENDIAN_BIG)
			reexecuteData = (reexecuteData || byteValue != 0)? true : false;   
#endif
			triangleFlipsDst[i*4 + b] = byteValue;
		}
	}

	return reexecuteData;
}

static void hclUpdateAllVertexFramesOperator_2_to_3(hkDataObject& obj)
{
	hkDataArray triangleFlipsSrc = obj["old_triangleFlips"].asArray();
	hkDataArray triangleFlipsDst = obj["triangleFlips"].asArray();

	bool reexecuteClothData = udpateTriangleFlips(triangleFlipsSrc, triangleFlipsDst);
	if(reexecuteClothData)
	{
		HK_WARN_ALWAYS(0xabbaee55, "hclUpdateAllVertexFramesOperator previous to version 3 had potentially incorrect data in m_triangleFlips on big-endian platforms. If the cloth-data was generated using a big-endian platform then its associated cloth setup data should be re-executed.");
	}
}

static void hclUpdateSomeVertexFramesOperator_2_to_3(hkDataObject& obj)
{
	hkDataArray triangleFlipsSrc = obj["old_triangleFlips"].asArray();
	hkDataArray triangleFlipsDst = obj["triangleFlips"].asArray();

	bool reexecuteClothData = udpateTriangleFlips(triangleFlipsSrc, triangleFlipsDst);
	if(reexecuteClothData)
	{
		HK_WARN_ALWAYS(0xabba86fc, "hclUpdateSomeVertexFramesOperator previous to version 3 had potentially incorrect data in m_triangleFlips on big-endian platforms. If the cloth-data was generated using a big-endian platform then its associated cloth setup data should be re-executed.");
	}
}

static void hclSimClothData_9_to_10(hkDataObject& obj)
{
	hkDataArray triangleFlipsSrc = obj["old_triangleFlips"].asArray();
	hkDataArray triangleFlipsDst = obj["triangleFlips"].asArray();
	
	bool reexecuteClothData = udpateTriangleFlips(triangleFlipsSrc, triangleFlipsDst);
	if(reexecuteClothData)
	{
		HK_WARN_ALWAYS(0xabbabb9c, "hclSimClothData previous to version 10 had potentially incorrect data in m_triangleFlips on big-endian platforms. If the cloth-data was generated using a big-endian platform then its associated cloth setup data should be re-executed.");
	}
}

// Registration function is at the end of the file

void HK_CALL registerClothPatches_2013_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2013_1/hclPatches_2013_1.cxx>
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
