/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Container/BitField/hkBitField.h>

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/Modifiers/hkpRemoveTerminalsMoppModifier.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppMachine.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>

static inline hkUchar getMoppInstruction (const hkUint32 removeInfo)
{
	return hkUchar(removeInfo & 0x000000FF); // Return the lower 8 buts
}

static inline hkUint32 getMoppOffset(const hkUint32 removeInfo)
{
	return (removeInfo & 0xFFFFFF00) >> 8; // Return the higher 24 bits, shifted down
}

static inline void setMoppInstruction (hkUint32& removeInfo, const hkUchar instruction)
{
	removeInfo &= 0xFFFFFF00;	// Clear the lower 8 bits
	removeInfo |= instruction;	// Insert the instruction
}

static inline void setMoppOffset(hkUint32& removeInfo, const hkUint32 offset)
{
	const hkUint32 shiftedOffset = (offset << 8);
	removeInfo &= 0x000000FF;		// Clear the upper 24 bits
	removeInfo |= shiftedOffset;	// Insert the offset
}

hkpRemoveTerminalsMoppModifier::hkpRemoveTerminalsMoppModifier( const hkpMoppCode* moppCode, const hkpShapeContainer* shapeContainer, const hkArray<hkpShapeKey>& shapesToRemove )
{
	//
	//	Calc the AABB of all nodes
	//
	hkAabb aabb;
	{
		const hkReal tolerance = 0.0f;

		hkpShapeBuffer shapeBuffer;

		if ( !shapesToRemove.getSize() )
		{
			return;
		}
		{
			hkpShapeKey key = shapesToRemove[0];
			const hkpShape* childShape = shapeContainer->getChildShape( key, shapeBuffer );
			childShape->getAabb( hkTransform::getIdentity(), tolerance, aabb );
		}
		
		for ( int i = 1; i < shapesToRemove.getSize(); i++)
		{
			hkpShapeKey key = shapesToRemove[i];
			const hkpShape* childShape = shapeContainer->getChildShape( key, shapeBuffer );

			hkAabb localAabb;
			childShape->getAabb(  hkTransform::getIdentity(), tolerance, localAabb );
			aabb.m_min.setMin( aabb.m_min, localAabb.m_min );
			aabb.m_max.setMax( aabb.m_max, localAabb.m_max );
		}
	}

	m_tempShapesToRemove = &shapesToRemove;
	hkMoppModifyVirtualMachine_queryAabb( moppCode, aabb, this);
	m_tempShapesToRemove = HK_NULL;
}

hkpRemoveTerminalsMoppModifier::hkpRemoveTerminalsMoppModifier( const hkpMoppCode* moppCode, const hkAabb& aabb, const hkArray<hkpShapeKey>& shapesToRemove )
{
	m_tempShapesToRemove = &shapesToRemove;
	hkMoppModifyVirtualMachine_queryAabb( moppCode, aabb, this);
	m_tempShapesToRemove = HK_NULL;
}

hkpRemoveTerminalsMoppModifier::~hkpRemoveTerminalsMoppModifier()
{

}

void hkpRemoveTerminalsMoppModifier::applyRemoveTerminals( hkpMoppCode* moppCode )
{
	for (int i = 0; i < m_removeInfo.getSize(); i++)
	{
		hkUchar* program = const_cast<hkUchar*>(&moppCode->m_data[0]) + getMoppOffset(m_removeInfo[i]);
		setMoppInstruction(m_removeInfo[i], *program);
		*program = 0;
	}
}

void hkpRemoveTerminalsMoppModifier::undoRemoveTerminals( hkpMoppCode* moppCode )
{
	for (int i = 0; i < m_removeInfo.getSize(); i++)
	{
		hkUchar* program = const_cast<hkUchar*>(&moppCode->m_data[0]) + getMoppOffset(m_removeInfo[i]);
		HK_ASSERT2(0x317ca32c,  *program == 0, "Inconsistent use of undoRemoveTerminals. Modifiers must be 'undone' in the reverse order to that in which they were 'applied', or perhaps this modifier's undo has been called twice?" );
		*program = getMoppInstruction(m_removeInfo[i]);
	}
}

			// hkpMoppModifier interface implementation
hkBool hkpRemoveTerminalsMoppModifier::shouldTerminalBeRemoved( hkUint32 id, const hkUint32  *properties )
{
	int find = m_tempShapesToRemove->indexOf( id );
	return find >=0;
}

			// hkpMoppModifier interface implementation
void hkpRemoveTerminalsMoppModifier::addTerminalRemoveInfo( hkInt32 relativeMoppAddress )
{
	hkUint32& rm = m_removeInfo.expandOne();
	setMoppInstruction(rm, 0);
	setMoppOffset(rm, relativeMoppAddress);
}


// hkpMoppModifier interface implementation
hkBool hkpRemoveTerminalsMoppModifier2::shouldTerminalBeRemoved( hkUint32 id, const hkUint32  *properties )
{
	int find = m_enabledKeys->get( id );
	return find == 0;
}


hkpRemoveTerminalsMoppModifier2::hkpRemoveTerminalsMoppModifier2( const hkpMoppCode* moppCode, const hkBitField& enabledKeys )
{
	m_enabledKeys = &enabledKeys;
	hkAabb aabb;
	aabb.m_max.setAll(  1e10f );
	aabb.m_min.setAll( -1e10f );
	hkMoppModifyVirtualMachine_queryAabb( moppCode, aabb, this);
	m_enabledKeys = HK_NULL;
}

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
