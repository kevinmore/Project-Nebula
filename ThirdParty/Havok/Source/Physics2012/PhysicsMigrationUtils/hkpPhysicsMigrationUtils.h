/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKP_PHYSICS_MIGRATION_UTILS_H
#define HKP_PHYSICS_MIGRATION_UTILS_H

#include <Common/Base/hkBase.h>
#include <Common/Base/KeyCode.h>

#if !defined(HK_FEATURE_PRODUCT_PHYSICS) || !defined(HK_FEATURE_PRODUCT_PHYSICS_2012)
	#error Physics and Physics 2012 are needed to use this library.
#endif

#include <Common/Base/Container/PointerMap/hkPointerMap.h>

class hkRootLevelContainer;
class hkpShape;
class hkpRigidBody;
class hkpConstraintInstance;
class hkpPhysicsSystem;
class hknpShape;
struct hknpBodyId;
struct hknpBodyCinfo;
class hknpConstraintCinfo;
class hknpPhysicsSystemData;
class hknpWorld;


/// Some conversion utilities for migrating from Physics 2012 to Physics.
/// These functions have been made to support specific use cases in our demos and product SDK.
/// As such, all use cases haven't been covered.
namespace hkpPhysicsMigrationUtils
{
	/// Convert from old to new shape.
	hknpShape* HK_CALL convertShape( const hkpShape& physicsShape );

	/// Convert from old to new rigid body (cinfo).
	hknpBodyCinfo* HK_CALL convertBody(
		const hkpRigidBody& body,
		hkVector4Parameter gravity,
		hknpPhysicsSystemData& systemDataOut,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap = HK_NULL );

	/// Convex from old to new constraint (cinfo).
	void HK_CALL convertConstraint(
		const hkpConstraintInstance& constraint,
		hknpConstraintCinfo& constraintCinfoOut,
		const hkArrayBase<hkpRigidBody*>& bodyMap );

	/// Convert from old to new physics system.
	/// Note: Use hkaPhysicsMigrationUtils to convert ragdolls.
	void HK_CALL convertPhysicsSystem(
		const hkpPhysicsSystem& system,
		hknpPhysicsSystemData& systemDataOut,
		const hkVector4* newWorldGravity = HK_NULL,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap = HK_NULL );

	/// Convert physics data found in rootLevelContainer and put the data in rootLevelContainerOut.
	/// If these are the same container, the old data isn't removed and the new data is given a converted name.
	/// If these are different containers, rootLevelContainerOut will point to non-physics data contained in rootLevelContainer; it is not copied.
	/// Additionally if pruneNonConvertedPhysicsData is true, the data inside rootLevelContainer may be affected as the util doesn't make copies.
	/// Returns if the output differs from the input.
	bool HK_CALL convertRootLevelContainer(
		const hkRootLevelContainer& rootLevelContainer,
		hkRootLevelContainer& rootLevelContainerOut,
		const hkVector4* newWorldGravity = HK_NULL,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap = HK_NULL,
		bool pruneNonConvertedPhysicsData = true );

	/// Add an old rigid body to a new physics world.
	hknpBodyId HK_CALL addBody(
		hknpWorld& world,
		const hkpRigidBody& body,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap = HK_NULL,
		bool forceKeyframed = false );

	/// Remove a body from a new physics world from it's old physics pointer.
	void HK_CALL removeBody(
		hknpWorld& world,
		const hkpRigidBody& body,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap = HK_NULL );

	/// Used by other conversion utils to prune deprecated classes.
	/// Returns if data is pruned
	bool HK_CALL pruneDeprecatedClasses(hkRootLevelContainer& rootLevelContainer);

	/// Because some hkp class names are still used by physics, it's not sufficient to simply check for an hkp prefix.
	/// This function will return if the provided class name is used only by Physics2012.
	bool HK_CALL isDeprecatedClass( const char* className );
}

#endif // HKP_PHYSICS_MIGRATION_UTILS_H

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
