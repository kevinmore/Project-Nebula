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

static void hkpDeformableLinConstraintAtom_0_to_1(hkDataObject& obj)
{
	const hkReal* y = obj["yieldStrength"].asVec(8);
	obj["yieldStrengthDiag"].setVec(y,4);
	obj["yieldStrengthOffDiag"].setVec(y+4,4);

	const hkReal* u = obj["ultimateStrength"].asVec(8);
	obj["ultimateStrengthDiag"].setVec(u,4);
	obj["ultimateStrengthOffDiag"].setVec(u+4,4);
}

static void hkpDeformableAngConstraintAtom_0_to_1(hkDataObject& obj)
{
	const hkReal* y = obj["yieldStrength"].asVec(8);
	obj["yieldStrengthDiag"].setVec(y,4);
	obj["yieldStrengthOffDiag"].setVec(y+4,4);

	const hkReal* u = obj["ultimateStrength"].asVec(8);
	obj["ultimateStrengthDiag"].setVec(u,4);
	obj["ultimateStrengthOffDiag"].setVec(u+4,4);
}

static void hkpConvexVerticesShape_4_to_5(hkDataObject& obj)
{
	hkDataArray src = obj["rotatedVertices_old"].asArray(); // array of hkFourTransposedPoints
	hkDataArray dst = obj["rotatedVertices"].asArray();     // array of 12*hkReal

	const int size = src.getSize();
	dst.setSize(12*size);
	for (int i = 0; i < size; i++)
	{
		hkDataObject src_obj = src[i].asObject(); // one hkFourTransposedPoints
		hkDataArray src_i = src_obj["vertices"].asArray();

		{
			const hkVector4& vec = src_i[0].asVector4();
			dst[i*12  ] = hkReal(vec(0));
			dst[i*12+1] = hkReal(vec(1));
			dst[i*12+2] = hkReal(vec(2));
			dst[i*12+3] = hkReal(vec(3));
		}
		{
			const hkVector4& vec = src_i[1].asVector4();
			dst[i*12+4] = hkReal(vec(0));
			dst[i*12+5] = hkReal(vec(1));
			dst[i*12+6] = hkReal(vec(2));
			dst[i*12+7] = hkReal(vec(3));
		}
		{
			const hkVector4& vec = src_i[2].asVector4();
			dst[i*12+8] = hkReal(vec(0));
			dst[i*12+9] = hkReal(vec(1));
			dst[i*12+10] = hkReal(vec(2));
			dst[i*12+11] = hkReal(vec(3));
		}
	}
}

static void hkpConvexVerticesShape_5_to_6(hkDataObject& obj)
{
	hkDataArray src = obj["rotatedVertices_old"].asArray(); // array of 12*hkReal
	hkDataArray dst = obj["rotatedVertices"].asArray();     // array of hkMatrix3

	const int size = src.getSize()/12;
	dst.setSize(size);
	for (int i = 0; i < size; i++)
	{
		hkMatrix3 dst_obj;
		hkVector4 vec;
		{
			vec.set(src[i*12+0].asReal(), src[i*12+1].asReal(), src[i*12+2].asReal(), src[i*12+3].asReal());
			dst_obj.setColumn<0>(vec);
		}
		{
			vec.set(src[i*12+4].asReal(), src[i*12+5].asReal(), src[i*12+6].asReal(), src[i*12+7].asReal());
			dst_obj.setColumn<1>(vec);
		}
		{
			vec.set(src[i*12+8].asReal(), src[i*12+9].asReal(), src[i*12+10].asReal(), src[i*12+11].asReal());
			dst_obj.setColumn<2>(vec);
		}
		dst[i] = dst_obj; // one hkMatrix3
	}
}

void HK_CALL registerPhysicsPatches_2012_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2012_2/hkpPatches_2012_2.cxx>
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
