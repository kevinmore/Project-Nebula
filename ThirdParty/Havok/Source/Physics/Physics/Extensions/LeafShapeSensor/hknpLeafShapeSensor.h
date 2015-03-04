/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_LEAF_SHAPE_SENSOR_H
#define HKNP_LEAF_SHAPE_SENSOR_H

#include <Common/Base/hkBase.h>
#include <Common/Base/Container/PointerMap/hkMap.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>

#include <Physics/Physics/hknpTypes.h>


/// A utility which caches the set of leaf shapes which overlap a world space AABB,
/// and reports any changes in that set to the user.
class hknpLeafShapeSensor : public hkReferencedObject
{
	public:

		// A unique identifier for a leaf shape in the world.
		struct LeafShapeId
		{
			LeafShapeId() {}
			LeafShapeId( const LeafShapeId& id ) : m_bodyId(id.m_bodyId), m_shapeKey(id.m_shapeKey) {}
			LeafShapeId( const hknpBodyId bodyId, const hknpShapeKey shapeKey ) : m_bodyId(bodyId), m_shapeKey(shapeKey) {}

			hknpBodyId		m_bodyId;
			hknpShapeKey	m_shapeKey;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor.
		hknpLeafShapeSensor( hknpWorld* world, hkInt32 capacity = 1024 );

		/// Destructor.
		virtual ~hknpLeafShapeSensor();

	public:
		/// Set the AABB expansion margin, used for static collision to make sure that the AABB is only re-queried if
		/// the current AABB breaks out of the expanded one (the expanded one persists from last query).
		void	setAabbExpansionMargin( hkReal margin );

		/// Set the maximum number of leaf shapes that can be collected.
		void	setCapacity( hkInt32 capacity );

		/// Update the AABB and call the callback methods for any changes in the leaf shape set.
		void	updateAabb( hkAabb& aabbIn );

		/// Get user data.
		hkUlong getUserData( const LeafShapeId& id );

		/// Remove the context associated with the bodyId in m_bodyContexts
		void	removeContext( const hknpBodyId& bodyId );

	public:
		// Whether collision detection is enabled for this body
		virtual bool	isCollisionEnabled( const hknpBodyId& bodyId ) { return true; }

		/// Whether to only collide with the body and don't process it's leaf shapes
		virtual bool	collideWithBodyOnly( const hknpBodyId& bodyId ) { return false; }

		//
		// Callback handlers, must be overloaded
		//

		/// The aabb to use when colliding with a static/dynamic body
		virtual void	getAaab( bool bodyIsStatic, hkAabb& aabb ) = 0;

		/// Called if there was any change in the set.
		virtual void	shapesUpdatedCallback() = 0;

		/// Called upon enter Callback. You should return your user data here.
		virtual hkUlong shapeEnteredCallback( const LeafShapeId& id ) = 0;

		/// Called when a leaf shape has changed transform or other properties.
		virtual void	shapeUpdatedCallback( const LeafShapeId& id, hkUlong data ) = 0;

		/// Called when shape exits aabb, including user data.
		virtual void	shapeLeftCallback( const LeafShapeId& id, const hkUlong data ) = 0;

		/// Called upon enter Callback for a body whose leaf shapes are ignored. You should return your user data here.
		virtual void	bodyUpdatedCallback( const LeafShapeId& id, hkUlong data ) = 0;

		/// Called when a body whose leaf shapes are ignored has changed transform or other properties.
		virtual hkUlong bodyEnteredCallback( const LeafShapeId& id ) = 0;

		/// Called when a body whose leaf shapes are ignored exits aabb, including user data.
		virtual void	bodyLeftCallback( const LeafShapeId& id, const hkUlong userInfo ) = 0;

	protected:

		//
		// Internal data structure types
		//

		enum CallbackType
		{
			ENTERING,
			STAYING,
			LEAVING
		};

		//
		template <typename USER> struct Data
		{
			Data() {}
			USER						m_userData;
			hkEnum<CallbackType, hkUint8>	m_callbackType;
		};

		//
		struct Context
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, Context );

			Context( hknpBodyId bodyId, hkBool isStatic, hkBool isComposite, hkBool collideWithBodyOnly )
				: m_bodyId(bodyId), m_isStatic(isStatic), m_isComposite(isComposite), m_collideWithBodyOnly(collideWithBodyOnly),
					m_userData(0), m_shapeKeys(new hkMap<hknpShapeKey,Data<hkUlong> >() )
			{
				m_aabb.setEmpty();
			}

			~Context()
			{
				delete m_shapeKeys;
			}

			hkAabb									m_aabb;
			hknpBodyId								m_bodyId;
			hkBool									m_isStatic;
			hkBool									m_isComposite;
			hkBool									m_collideWithBodyOnly;		// Leaf shapes are ignored
			hkUlong									m_userData;
			hkMap<hknpShapeKey, Data<hkUlong> >*	m_shapeKeys;
		};

		// Hash map functions for hknpBodyId
		struct MapOpsBodyId : public hkMapOperations<hknpBodyId>
		{
			inline static unsigned	hash( hknpBodyId key, unsigned mod ) { return (unsigned(key.value()) * 2654435761U) & mod; }
			inline static void		invalidate( hknpBodyId& key ) { new ( /*LLVM hack*/reinterpret_cast<hkPlacementNewArg*>(&key)) hknpBodyId(hknpBodyId::InvalidValue); }
			inline static hkBool32	isValid( hknpBodyId key ) { return key.isValid(); }
			inline static hkBool32	equal( hknpBodyId key0, hknpBodyId key1 ) { return (key0.value() == key1.value()); }
		};

	protected:

		//
		// Internal methods
		//

		Context*	createContext( const hkAabb& aabb, const hknpBodyId& bodyId );
		void		processContext( const hkAabb& aabb, Context* context );
		void		processContextComposite( const hkAabb& aabb, Context* context );
		void		removeContext( Context* context );

		HK_FORCE_INLINE Context*	bodyEntered( const hkAabb& aabb, const hknpBodyId& bodyId);
		HK_FORCE_INLINE void		bodyLeft( const hkAabb& aabb, const hknpBodyId& bodyId, Context* const context );
		HK_FORCE_INLINE Context*	bodyStayed( const hkAabb& aabb, const hknpBodyId& bodyId, Context* const context );
		HK_FORCE_INLINE void		bodyCleaup( const hkAabb& aabb, const hknpBodyId& bodyId, Context* const context );

		HK_FORCE_INLINE hkUlong		childShapeEntered( const Context* context, const hknpShapeKey& shapeKey );
		HK_FORCE_INLINE void		childShapeLeft( const Context* context, const hknpShapeKey& shapeKey, hkUlong user );
		HK_FORCE_INLINE hkUlong		childShapeStayed( const Context* context, const hknpShapeKey& shapeKey, hkUlong user );

		template <typename KEY, typename VALUE, typename OPS,
					typename ARRAY, typename CONTEXT, typename TYPE,
						typename CALLBACK1, typename CALLBACK2, typename CALLBACK3 >
		static HK_FORCE_INLINE void updateSet( hkMap<KEY,hknpLeafShapeSensor::Data<VALUE>,OPS>& keys, ARRAY& observedKeys, CONTEXT& context, TYPE& object,
													const CALLBACK1 enterCallback, const CALLBACK2 stayCallback, const CALLBACK3 leaveCallback );

		template <typename KEY, typename VALUE, typename OPS, typename CONTEXT, typename TYPE, typename CALLBACK3 >
		static HK_FORCE_INLINE void clearSet( hkMap<KEY,hknpLeafShapeSensor::Data<VALUE>,OPS>& keys, CONTEXT& context, TYPE& object, const CALLBACK3 leaveCallback );

	protected:

		hkInt32		m_capacity;
		hknpWorld*	m_world;
		hkReal		m_aabbExpansionMargin;
		hkMap< hknpBodyId, Data<Context*>, MapOpsBodyId> m_bodyContexts;
};

#endif	// HKNP_LEAF_SHAPE_SENSOR_H

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
