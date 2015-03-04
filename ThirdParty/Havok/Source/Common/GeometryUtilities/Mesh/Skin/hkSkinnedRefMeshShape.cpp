/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Skin/hkSkinnedRefMeshShape.h>
#include <Common/Base/Reflection/hkClass.h>

//
//	Constructor

hkSkinnedRefMeshShape::hkSkinnedRefMeshShape(hkSkinnedMeshShape* fullSkin, const short* bones, const hkQTransform* localFromRootTransforms, int numBones)
:	hkMeshShape()
,	m_skinnedMeshShape(fullSkin)
,	m_name(HK_NULL)
{
	m_bones.append(bones, numBones);

	const hkVector4* srcPtr = reinterpret_cast<const hkVector4*>(localFromRootTransforms);
	hkVector4* dstPtr = m_localFromRootTransforms.expandBy(numBones*2);
	for (int i = 0; i < numBones*2; i++)
	{
		*dstPtr++ = *srcPtr++;
	}
}

//
//	Constructor

hkSkinnedRefMeshShape::hkSkinnedRefMeshShape(hkSkinnedMeshShape* fullSkin)
:	hkMeshShape()
,	m_skinnedMeshShape(fullSkin)
,	m_name(HK_NULL)
{}

//
//	Serialization constructor

hkSkinnedRefMeshShape::hkSkinnedRefMeshShape(class hkFinishLoadedObjectFlag flag)
:	hkMeshShape(flag)
,	m_skinnedMeshShape(flag)
,	m_bones(flag)
,	m_localFromRootTransforms(flag)
,	m_name(flag)
{}

//
//	Destructor

hkSkinnedRefMeshShape::~hkSkinnedRefMeshShape()
{
	m_skinnedMeshShape = HK_NULL;
	m_bones.clearAndDeallocate();
}

//
//	Creates a compound mesh by merging all the given Vision meshes

hkSkinnedRefMeshShape* HK_CALL hkSkinnedRefMeshShape::create(const hkMeshShape*const* shapes, const hkQTransform* transforms, int numShapes)
{
	hkSkinnedMeshShape* fullSkin = HK_NULL;
	int numBones = 0;

	// Count the number of bones and make sure all shapes share the same skin
	for (int k = 0; k < numShapes; k++)
	{
		// Check if we were given an actual skin ref
		const hkMeshShape* shapeBase =shapes[k];
		if ( !hkSkinnedRefMeshShapeClass.equals(shapeBase->getClassType()) )
		{
			HK_ASSERT(0x293dfaf1, false);
			return HK_NULL;	// Should not happen, we can only create a skinned mesh shape out of skin refs!
		}

		// Check if the skin ref is for the same skin
		const hkSkinnedRefMeshShape* skinRef = reinterpret_cast<const hkSkinnedRefMeshShape*>(shapeBase);
		if ( fullSkin && (fullSkin != skinRef->m_skinnedMeshShape) )
		{
			HK_ASSERT(0x293dfaf1, false);
			return HK_NULL;	// Should not happen, we can only create a skinned mesh shape out of skin refs pointing to the same skin!
		}

		fullSkin  = skinRef->m_skinnedMeshShape;
		numBones += skinRef->m_bones.getSize();
	}

	// Create the compound shape
	hkSkinnedRefMeshShape* compound = new hkSkinnedRefMeshShape(fullSkin);

	// Set-up the bones
	hkArray<short>& dstBones = compound->m_bones;
	dstBones.setSize(numBones);
	compound->m_localFromRootTransforms.setSize(numBones*2);
	hkQTransform* dstTransforms = reinterpret_cast<hkQTransform*>(compound->m_localFromRootTransforms.begin());
	for (int k = 0, boneIndex = 0; k < numShapes; k++)
	{
		const hkSkinnedRefMeshShape* skinRef = reinterpret_cast<const hkSkinnedRefMeshShape*>(shapes[k]);
		const hkArray<short>& srcBones = skinRef->m_bones;
		const hkQTransform* srcTransforms = reinterpret_cast<const hkQTransform*>(skinRef->m_localFromRootTransforms.begin());

		const int numShapeBones = skinRef->m_bones.getSize();
		for (int bi = 0; bi < numShapeBones; bi++, boneIndex++)
		{
			dstBones[boneIndex] = srcBones[bi];
			dstTransforms[boneIndex].setMul(transforms[k], srcTransforms[bi]);
		}
	}

	// Return the newly created compound shape
	return compound;
}

//
//	Returns the amount of sections

int hkSkinnedRefMeshShape::getNumSections() const
{
	// Should never be called!
	return 0;
}

//
//	Gets information about a section

void hkSkinnedRefMeshShape::lockSection(int sectionIndex, hkUint8 accessFlags, hkMeshSection& sectionOut) const
{
	// Should never be called!
	HK_ASSERT(0x1b11dab5, false);
}

//
//	Unlocks a mesh section. Must be given exactly the same structure contents as was returned from a lockSection
//	otherwise behavior is undefined.

void hkSkinnedRefMeshShape::unlockSection(const hkMeshSection& section) const
{
	// Should never be called!
	HK_ASSERT(0x1b11dab5, false);
}

//
//	Returns an optional name of the mesh shape

const char* hkSkinnedRefMeshShape::getName() const
{
	return m_name;
}

//
//	Sets an optional name

void hkSkinnedRefMeshShape::setName(const char* n)
{
	m_name = n;
}

//
//	Returns the class type

const hkClass* hkSkinnedRefMeshShape::getClassType() const
{
	return &hkSkinnedRefMeshShapeClass;
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
