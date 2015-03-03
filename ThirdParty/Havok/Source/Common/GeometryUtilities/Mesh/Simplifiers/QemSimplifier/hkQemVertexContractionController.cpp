/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>

// this
#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkQemVertexContractionController.h>

hkBool32 hkQemVertexContractionController::allowContraction(const EdgeContraction& contraction)
{
	switch (contraction.m_type)
	{
		case EdgeContraction::TYPE_SELECT_START:
		{
			return canVertexContract(contraction.m_end);
		}
		case EdgeContraction::TYPE_SELECT_END:
		{
			return canVertexContract(contraction.m_start);
		}
		case EdgeContraction::TYPE_NEW:
		{
			return canVertexContract(contraction.m_start) && canVertexContract(contraction.m_end);
		}
		default: break;
	}

	return true;
}

void hkQemVertexContractionController::setVertexCanContract(int index, bool enable)
{
	const int word = index / 32;
	const hkUint32 bit = hkUint32(1) << (index & 31);

	if (enable)
	{
		// See if it is in range
		if (word < m_bitField.getSize())
		{
			// Clear the bit
			m_bitField[word] &= ~bit;
		}
	}
	else
	{
		if (word >= m_bitField.getSize())
		{
			m_bitField.setSize(word + 1, 0);
		}
		m_bitField[word] |= bit;
	}
}

bool hkQemVertexContractionController::canVertexContract(int index) const
{
	const int word = index / 32;
	const hkUint32 bit = hkUint32(1) << (index & 31);

	if (word >= m_bitField.getSize())
	{
		return true;
	}

	return (m_bitField[word] & bit) == 0;
}

void hkQemVertexContractionController::setVertexCanContractFromAabb(const hkArray<hkVector4>& positions, const hkAabb& aabb, bool value)
{
	const int numPositions = positions.getSize();
	for (int i = 0; i < numPositions; i++)
	{
		const hkVector4& pos = positions[i];

		if (aabb.containsPoint(pos))
		{
			setVertexCanContract(i, value);
		}
	}
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
