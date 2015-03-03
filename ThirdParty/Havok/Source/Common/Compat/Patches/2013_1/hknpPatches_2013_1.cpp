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

extern void HK_CALL upgradeBitField_2013_1(hkDataObject& dstBitField, hkDataObject& srcBitField);

static hkUint32 HK_CALL convertBodyFlags_2013_1(const hkUint32& srcFlags)
{
	// Old body flag 8 has been removed, so we have to shift downwards all the others

	// Get the first 8 bits
	hkUint32 dstFlags = srcFlags & 0x000000ff;
	// And the next bits shifted down by one
	hkUint32 upperPart = ((srcFlags & 0xffffff00) >> 9) << 8;
	dstFlags |= (upperPart);
	return dstFlags;
}

static void hknpCompressedMeshShape_0_to_1(hkDataObject& obj)
{
	{
		hkDataObject srcBitField = obj["old_quadIsFlat"].asObject();
		hkDataObject dstBitField = obj["quadIsFlat"].asObject();
		upgradeBitField_2013_1(dstBitField, srcBitField);
	}
	{
		hkDataObject srcBitField = obj["old_trianglesWithDuplicatedEdges"].asObject();
		hkDataObject dstBitField = obj["trianglesWithDuplicatedEdges"].asObject();
		upgradeBitField_2013_1(dstBitField, srcBitField);
	}
}

static void HK_CALL hknpBody_0_to_1(hkDataObject& obj)
{
	obj["flags"] = (int) convertBodyFlags_2013_1( obj["flags"].asInt());
}

static void HK_CALL hknpBodyCInfo_0_to_1(hkDataObject& obj)
{
	obj["flags"] = (int) convertBodyFlags_2013_1( obj["flags"].asInt());
}

static void HK_CALL hknpConvexPolytopeShape_0_to_1(hkDataObject& obj)
{
	// Copy of hknpShape::FlagsEnum version 1
	enum ShapeFlags_1
	{
		IS_CONVEX_SHAPE					= 1<<0,
		IS_CONVEX_POLYTOPE_SHAPE		= 1<<1,
		IS_COMPOSITE_SHAPE				= 1<<2,
		IS_HEIGHT_FIELD_SHAPE			= 1<<3,
		USE_SINGLE_POINT_MANIFOLD		= 1<<4,
		IS_TRIANGLE_OR_QUAD_NO_EDGES	= 1<<5,
		SUPPORTS_BPLANE_COLLISIONS		= 1<<6,
		USE_NORMAL_TO_FIND_SUPPORT_PLANE= 1<<7,

		// These are new in version 1
		USE_SMALL_FACE_INDICES			= 1<<8,
		NO_GET_ALL_SHAPE_KEYS_ON_SPU	= 1<<9,
		SHAPE_NOT_SUPPORTED_ON_SPU		= 1<<10,
	};

	int flags = obj["flags"].asInt();
	int numFaces = obj["faces"].asArray().getSize();
	if (numFaces < 256)
	{
		obj["flags"] = (hkUint16) (flags | USE_SMALL_FACE_INDICES);
	}
	else
	{
		obj["flags"] = (hkUint16) (flags & ~USE_SMALL_FACE_INDICES);
	}
}



void HK_CALL registerNewPhysicsPatches_2013_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2013_1/hknpPatches_2013_1.cxx>
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
